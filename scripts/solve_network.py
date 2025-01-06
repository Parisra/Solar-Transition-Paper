# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""
import importlib
import logging
import os
import re
import sys

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _benchmark import memory_logger
from _helpers import configure_logging, get_opt, update_config_with_sector_opts
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pathlib import Path
from functools import reduce


logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)

def get(item, investment_year=None):
    """
    Check whether item depends on investment year.
    """
    return item[investment_year] if isinstance(item, dict) else item

def add_land_use_constraint(n, planning_horizons, config):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n, planning_horizons, config)
    else:
        _add_land_use_constraint(n)


def add_land_use_constraint_perfect(n):
    """
    Add global constraints for tech capacity limit.
    """
    logger.info("Add land-use constraint for perfect foresight")

    def compress_series(s):
        def process_group(group):
            if group.nunique() == 1:
                return pd.Series(group.iloc[0], index=[None])
            else:
                return group

        return s.groupby(level=[0, 1]).apply(process_group)

    def new_index_name(t):
        # Convert all elements to string and filter out None values
        parts = [str(x) for x in t if x is not None]
        # Join with space, but use a dash for the last item if not None
        return " ".join(parts[:2]) + (f"-{parts[-1]}" if len(parts) > 2 else "")

    def check_p_min_p_max(p_nom_max):
        p_nom_min = n.generators[ext_i].groupby(grouper).sum().p_nom_min
        p_nom_min = p_nom_min.reindex(p_nom_max.index)
        check = (
            p_nom_min.groupby(level=[0, 1]).sum()
            > p_nom_max.groupby(level=[0, 1]).min()
        )
        if check.sum():
            logger.warning(
                f"summed p_min_pu values at node larger than technical potential {check[check].index}"
            )

    grouper = [n.generators.carrier, n.generators.bus, n.generators.build_year]
    ext_i = n.generators.p_nom_extendable
    # get technical limit per node and investment period
    p_nom_max = n.generators[ext_i].groupby(grouper).min().p_nom_max
    # drop carriers without tech limit
    p_nom_max = p_nom_max[~p_nom_max.isin([np.inf, np.nan])]
    # carrier
    carriers = p_nom_max.index.get_level_values(0).unique()
    gen_i = n.generators[(n.generators.carrier.isin(carriers)) & (ext_i)].index
    n.generators.loc[gen_i, "p_nom_min"] = 0
    # check minimum capacities
    check_p_min_p_max(p_nom_max)
    # drop multi entries in case p_nom_max stays constant in different periods
    # p_nom_max = compress_series(p_nom_max)
    # adjust name to fit syntax of nominal constraint per bus
    df = p_nom_max.reset_index()
    df["name"] = df.apply(
        lambda row: f"nom_max_{row['carrier']}"
        + (f"_{row['build_year']}" if row["build_year"] is not None else ""),
        axis=1,
    )

    for name in df.name.unique():
        df_carrier = df[df.name == name]
        bus = df_carrier.bus
        n.buses.loc[bus, name] = df_carrier.p_nom_max.values

    return n


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "solar rooftop", "solar-delta","solar inv","solar-hsat","solar-hsat inv", "onwind", "offwind-ac", "offwind-dc"]:  ##added solar rooftop
        extendable_i = (n.generators.carrier == carrier) & n.generators.p_nom_extendable
        n.generators.loc[extendable_i, "p_nom_min"] = 0

        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n, planning_horizons, config):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    grouping_years = config["existing_capacities"]["grouping_years_power"] 
    current_horizon = snakemake.wildcards.planning_horizons

    #carrier_names=  {   ## added to fix issue with -inv carrier name not matching , TODO: fix the names for an easier solution
    #"solar":"solar", "solar rooftop":"solar rooftop" ,"solar-hsat":"solar-hsat" ,"solar-delta": "solar-delta",
    #"onwind":"onwind" , "offwind-ac": "offwind-ac", "offwind-dc":"offwind-dc",  
    #"solar-inv-1.7":"solar inv","solar-inv-1.9":"solar inv", "solar-hsat-inv-1.5":"solar-hsat inv", "solar-hsat-inv-1.9":"solar-hsat inv", 
    #"solar rooftop-inv-1.3":"solar rooftop","solar rooftop-inv-1.7":"solar rooftop","solar rooftop southeast-inv-1.5":"solar rooftop"
    #}

    rooftop_techs = list(set([' '.join(i.split(' ')[2:]) for i in n.generators.index[n.generators.carrier=='solar rooftop']]))
    rooftop_techs = ['-'.join(i.split('-')[:-1]) for i in rooftop_techs]  ## delete year
    delta_techs = list(set([' '.join(i.split(' ')[-1:]) for i in n.generators.index[n.generators.carrier=='solar-delta']]))
    delta_techs = ['-'.join(i.split('-')[:-1]) for i in delta_techs]  ## delete year
    carrier_list = [i for i in config['electricity']['renewable_carriers'] if i in n.generators.carrier.unique() and i!='hydro'] + rooftop_techs + delta_techs
   
    print('current horizon is ', current_horizon)
    print('reducing potential of previous years for:', carrier_list )

    for carrier in carrier_list : #carrier_list.keys():
        carrier_name = 'solar rooftop' if 'solar rooftop' in carrier else 'solar-delta' if 'solar-delta' in carrier else carrier
        existing = n.generators.loc[n.generators.carrier == carrier_name, "p_nom"]
        ind = list(
            {i.split(sep=" ")[0] + " " + i.split(sep=" ")[1] for i in existing.index}
        )


        previous_years = [  ##fixed duplicated years
            str(y)
            for y in set(planning_horizons + grouping_years)
            if y < int(snakemake.wildcards.planning_horizons)
        ]
        

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    # check if existing capacities are larger than technical potential  ##added
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                        adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]    

    n.generators.p_nom_max.clip(lower=0, inplace=True)

def add_solar_potential_constraints(n, config):  ##config
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, fixed with inverter, tracking, delta-shaped) is below the region potential.
    """
    if "m" in snakemake.wildcards.clusters:
        location = (
        pd.Series([' '.join(i.split(' ')[:2]) for i in n.generators.index], index=n.generators.index)
        )

        def group(df, b="bus"):

            return df[b].map(location).to_xarray()

        gen_index = n.generators[n.generators.p_nom_extendable].index
        gen_past = n.generators[~n.generators.p_nom_extendable].index

        ## filter all utility solar generation except solar thermal
        filters = [("solar", True), ("thermal", False), ("rooftop", False)]
        solar = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters, gen_index)
        solar_original= n.generators[(n.generators.carrier=='solar') & (n.generators.p_nom_extendable)].index
        solar_other = [i for i in reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]], 
                                         filters, gen_past) if i not in 
                       n.generators[(n.generators.carrier=='solar') & ~(n.generators.p_nom_extendable)].index]

        land_use_factors= {
        'solar-delta' : 0.73  ,                                                                     
        'solar-hsat'  : 1.15 ,                                                                      
        }
        land_use = pd.DataFrame(1, index=solar, columns=['land_use_factor'])
        for key in land_use_factors.keys():
            land_use = land_use.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x,  axis=1)

        dc_capacity = pd.DataFrame(1, index=solar, columns=['dc_capacity'])
        dc_capacity = dc_capacity.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if  ##TODO: it should also be able to do 1.25
                                'inv' in x.name else x,  axis=1)   
    
        #print(n.generators.bus.loc[solar].map(location))   
        #print(n.generators.bus.loc[solar].map(location).to_xarray())
        rename = {"Generator-ext": "Generator"}

        ggrouper= pd.Series( n.generators.loc[solar].index.rename('bus').map(location), index=n.generators.loc[solar].index,).to_xarray()
        lhs = (
            (n.model["Generator-p_nom"].rename(rename).loc[solar]
            *land_use.squeeze().values*dc_capacity.squeeze().values)
            .groupby(ggrouper) #.groupby(group(n.generators.loc[solar]))
            .sum()
        )    
        #print('solar:', lhs)
        ## using only the original solar  p_nom_max disregrads the installed capacity of other solar techs like hsat
        installed = pd.DataFrame(n.generators.p_nom_opt.loc[solar_other])
        for key in land_use_factors.keys():
            installed= installed.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x, axis=1)
        installed=installed.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                                'inv' in x.name else x,  axis=1).squeeze(axis=1).groupby(n.generators.loc[solar_other].index.rename('bus').map(location)).sum()
        #print('solar other', solar_other)
    
        rhs = (n.generators.loc[solar_original,"p_nom_max"]
                            .groupby(n.generators.loc[solar_original].index.rename('bus').map(location)).sum() ) 

        rhs = (rhs - installed.reindex(rhs.index).fillna(0)).clip(lower=0)  if  len(solar_other) > 0 else rhs.clip(lower=0)   
        print(lhs)
        print(rhs)
        n.model.add_constraints(lhs <= rhs, name="solar_potential")  

        print('adding solar rooftop constraints...')
        ## filter all rooftop solar generatos
        filters = [("solar", True), ("thermal", False), ("rooftop", True)]
        solar_rooftop = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters, gen_index)

        if config["foresight"] == "overnight":
            filters_ = [("solar rooftop", True), ("rooftop ", False),("inv", False), ]  
            filters__ = [("solar rooftop", True), ("rooftop ", False),("inv", False)] ##  rooftop capacity from previous years already subtracted in land constraints
        else:
            filters_ = [("solar rooftop", True), ("rooftop ", False),("inv", False), (snakemake.wildcards.planning_horizons,True)]  
            filters__ = [("solar rooftop", True), ("rooftop ", False),("inv", False), (snakemake.wildcards.planning_horizons,False)] ##  rooftop capacity from previous years already subtracted in land constraints
        
        rooftop_original = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters_, gen_index)
        rooftop_other = [i for i in 
                         reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]],  filters, gen_past) if i not in 
                         reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]], filters__, gen_past)] 

        ## in case alternates are ony considerd for utility or only rooftop are  (reduces a redundant constraint?)
        #if solar_new.empty:
        #   return

        land_use_factors= {
        'solar rooftop delta' : 0.73  ,                                                                                                                                          
       }
        land_use = pd.DataFrame(1, index=solar_rooftop, columns=['land_use_factor'])
        for key in land_use_factors.keys():
            land_use = land_use.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x,  axis=1)

        dc_capacity = pd.DataFrame(1, index=solar_rooftop, columns=['dc_capacity'])
        dc_capacity = dc_capacity.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                                'inv' in x.name else x,  axis=1)   
                                
        ggrouper= pd.Series(n.generators.loc[solar_rooftop].index.rename('bus').map(location), index=n.generators.loc[solar_rooftop].index).to_xarray()
        lhs = (
            (n.model["Generator-p_nom"].rename(rename).loc[solar_rooftop]
            *land_use.squeeze().values*dc_capacity.squeeze().values)
            .groupby(ggrouper)
            .sum()
        )    
        print(lhs)
        installed = pd.DataFrame(n.generators.p_nom_opt.loc[rooftop_other])
        for key in land_use_factors.keys():
          installed= installed.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x, axis=1)
        installed=installed.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                         'inv' in x.name else x,  axis=1).squeeze(axis=1).groupby(n.generators.loc[rooftop_other].index.rename('bus').map(location)).sum()
        rhs = (n.generators.loc[rooftop_original, "p_nom_max"].replace([np.inf, -np.inf], 0).
               groupby(n.generators.loc[rooftop_original].index.rename('bus').map(location)).sum()) 
        rhs = (rhs - installed.reindex(rhs.index).fillna(0)).clip(lower=0) if  len(rooftop_other) > 0 else rhs.clip(lower=0)   

        #print(rooftop_other) 
        print(rhs)
        n.model.add_constraints(lhs <= rhs, name="solar_rooftop_potential")     

    else : 
        location = (
            n.buses.location
            if "location" in n.buses.columns
            else pd.Series(n.buses.index, index=n.buses.index)
        )

        def group(df, b="bus"):

            return df[b].map(location).to_xarray()

        gen_index = n.generators[n.generators.p_nom_extendable].index
        gen_past = n.generators[~n.generators.p_nom_extendable].index

        ## filter all utility solar generation except solar thermal
        filters = [("solar", True), ("thermal", False), ("rooftop", False)]
        solar = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters, gen_index)
        solar_original= n.generators[(n.generators.carrier=='solar') & (n.generators.p_nom_extendable)].index
        solar_other = [i for i in reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]], 
                                         filters, gen_past) if i not in 
                       n.generators[(n.generators.carrier=='solar') & ~(n.generators.p_nom_extendable)].index]

        land_use_factors= {
        'solar-delta' : 0.73  ,                                                                     
        'solar-hsat'  : 1.15 ,                                                                      
        }
        land_use = pd.DataFrame(1, index=solar, columns=['land_use_factor'])
        for key in land_use_factors.keys():
            land_use = land_use.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x,  axis=1)

        dc_capacity = pd.DataFrame(1, index=solar, columns=['dc_capacity'])
        dc_capacity = dc_capacity.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                                'inv' in x.name else x,  axis=1)   
    
        #print(n.generators.bus.loc[solar].map(location))   
        #print(n.generators.bus.loc[solar].map(location).to_xarray())
        rename = {"Generator-ext": "Generator"}

        ggrouper= (n.generators.loc[solar].bus)
        lhs = (
            (n.model["Generator-p_nom"].rename(rename).loc[solar]
            *land_use.squeeze().values*dc_capacity.squeeze().values)
            .groupby(ggrouper) #.groupby(group(n.generators.loc[solar]))
            .sum()
        )    
        #print('solar:', lhs)
        ## using only the original solar  p_nom_max disregrads the installed capacity of other solar techs like hsat
        installed = pd.DataFrame(n.generators.p_nom_opt.loc[solar_other])
        for key in land_use_factors.keys():
            installed= installed.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x, axis=1)
        installed=installed.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                                'inv' in x.name else x,  axis=1).squeeze(axis=1).groupby(n.generators.loc[solar_other].bus).sum()
        #print('solar other', solar_other)
    
        rhs = (n.generators.loc[solar_original,"p_nom_max"]
                            .groupby(n.generators.loc[solar_original].bus).sum() ) 

        rhs = (rhs - installed.reindex(rhs.index).fillna(0)).clip(lower=0)  if  len(solar_other) > 0 else rhs.clip(lower=0)   
        print(lhs)
        print(rhs)
        n.model.add_constraints(lhs <= rhs, name="solar_potential")  

        print('adding solar rooftop constraints...')
        ## filter all rooftop solar generatos
        filters = [("solar", True), ("thermal", False), ("rooftop", True)]
        solar_rooftop = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters, gen_index)

        if config["foresight"] == "overnight":
            filters_ = [("solar rooftop", True), ("rooftop ", False),("inv", False), ]  
            filters__ = [("solar rooftop", True), ("rooftop ", False),("inv", False)] ##  rooftop capacity from previous years already subtracted in land constraints
        else:
            filters_ = [("solar rooftop", True), ("rooftop ", False),("inv", False), (snakemake.wildcards.planning_horizons,True)]  
            filters__ = [("solar rooftop", True), ("rooftop ", False),("inv", False), (snakemake.wildcards.planning_horizons,False)] ##  rooftop capacity from previous years already subtracted in land constraints
        
        rooftop_original = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters_, gen_index)
        rooftop_other = [i for i in 
                         reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]],  filters, gen_past) if i not in 
                         reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]], filters__, gen_past)] 

        ## in case alternates are ony considerd for utility or only rooftop are  (reduces a redundant constraint?)
        #if solar_new.empty:
        #   return

        land_use_factors= {
        'solar rooftop delta' : 0.73  ,                                                                                                                                          
       }
        land_use = pd.DataFrame(1, index=solar_rooftop, columns=['land_use_factor'])
        for key in land_use_factors.keys():
            land_use = land_use.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x,  axis=1)

        dc_capacity = pd.DataFrame(1, index=solar_rooftop, columns=['dc_capacity'])
        dc_capacity = dc_capacity.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                                'inv' in x.name else x,  axis=1) 
          
        ggrouper= (n.generators.loc[solar_rooftop].bus)
        lhs = (
            (n.model["Generator-p_nom"].rename(rename).loc[solar_rooftop]
            *land_use.squeeze().values*dc_capacity.squeeze().values)
            .groupby(ggrouper)
            .sum()
        )    
        #print('solar rooftop', lhs)
        installed = pd.DataFrame(n.generators.p_nom_opt.loc[rooftop_other])
        for key in land_use_factors.keys():
          installed= installed.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x, axis=1)
        installed=installed.apply(lambda x: (x* float(x.name[x.name.find('inv-')+4:x.name.find('inv-')+7])) if 
                         'inv' in x.name else x,  axis=1).squeeze(axis=1).groupby(n.generators.loc[rooftop_other].bus).sum()
        rhs = (n.generators.loc[rooftop_original, "p_nom_max"].replace([np.inf, -np.inf], 0).
               groupby(n.generators.loc[rooftop_original].bus).sum()) 
        rhs = (rhs - installed.reindex(rhs.index).fillna(0)).clip(lower=0) if  len(rooftop_other) > 0 else rhs.clip(lower=0)   

        print(lhs) 
        print(rhs)
        n.model.add_constraints(lhs <= rhs, name="solar_rooftop_potential") 

def add_wind_potential_constraints(n, config):  ##config
    """
    Add constraint to make sure the sum capacity of all solar technologies (fixed, fixed with inverter, tracking, delta-shaped) is below the region potential.
    """
    if "m" in snakemake.wildcards.clusters:
        location = (
        pd.Series([' '.join(i.split(' ')[:2]) for i in n.generators.index], index=n.generators.index)
        )

        def group(df, b="bus"):

            return df[b].map(location).to_xarray()

        gen_index = n.generators[n.generators.p_nom_extendable].index
        gen_past = n.generators[~n.generators.p_nom_extendable].index

        ## filter all utility solar generation except solar thermal
        filters = [("onwind", True), ]
        wind = reduce(lambda gen_index, f: gen_index[gen_index.str.contains(f[0]) == f[1]], filters, gen_index)
        wind_original= n.generators[(n.generators.carrier=='onwind') & (n.generators.p_nom_extendable)].index
        wind_other = [i for i in reduce(lambda gen_past, f: gen_past[gen_past.str.contains(f[0]) == f[1]], 
                                         filters, gen_past) if i not in 
                       n.generators[(n.generators.carrier=='onwind') & ~(n.generators.p_nom_extendable)].index]


        land_use = pd.DataFrame(1, index=wind, columns=['land_use_factor'])
        rename = {"Generator-ext": "Generator"}

        ggrouper= pd.Series( n.generators.loc[wind].index.rename('bus').map(location), 
                            index=n.generators.loc[wind].index,).to_xarray()
        lhs = (
            (n.model["Generator-p_nom"].rename(rename).loc[wind]
            *land_use.squeeze().values)
            .groupby(ggrouper) #.groupby(group(n.generators.loc[solar]))
            .sum()
        )    

        ## using only the original solar  p_nom_max disregrads the installed capacity of other solar techs like hsat
        installed = pd.DataFrame(n.generators.p_nom_opt.loc[wind_other])
        #for key in land_use_factors.keys():
         #   installed= installed.apply(lambda x: (x*land_use_factors[key]) if key in x.name else x, axis=1)
        installed=installed.groupby(n.generators.loc[wind_other].index.rename('bus').map(location)).sum()
    
        rhs = (n.generators.loc[wind_original,"p_nom_max"]
                 .groupby(n.generators.loc[wind_original].index.rename('bus').map(location)).sum() ) 

        rhs = (rhs) # - installed.reindex(rhs.index).fillna(0)).clip(lower=0)  if  len(wind_other) > 0 else rhs.clip(lower=0)   
        print(lhs)
        print(rhs)
        n.model.add_constraints(lhs <= rhs, name="wind_potential")  

        

def add_co2_sequestration_limit(n, config, limit=200):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """
    limit = limit * 1e6
    for o in opts:
        if "seq" not in o:
            continue
        limit = float(o[o.find("seq") + 3 :]) * 1e6
        break

    if not n.investment_periods.empty:
        periods = n.investment_periods
        names = pd.Index([f"co2_sequestration_limit-{period}" for period in periods])
    else:
        periods = [np.nan]
        names = pd.Index(["co2_sequestration_limit"])

    n.madd(
        "GlobalConstraint",
        names,
        sense=">=",
        constant=-limit,
        type="operational_limit",
        carrier_attribute="co2 sequestered",
        investment_period=periods,
    )


def add_carbon_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "co2_atmosphere"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            lhs = final_e.loc[time_i, :] - final_e.shift(snapshot=1).loc[time_i, :]

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_carbon_budget_constraint(n, snapshots):
    glcs = n.global_constraints.query('type == "Co2Budget"')
    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last = n.snapshot_weightings.reset_index().groupby("period").last()
            last_i = last.set_index([last.index, last.timestep]).index
            final_e = n.model["Store-e"].loc[last_i, stores.index]
            time_valid = int(glc.loc["investment_period"])
            time_i = pd.IndexSlice[time_valid, :]
            weighting = n.investment_period_weightings.loc[time_valid, "years"]
            lhs = final_e.loc[time_i, :] * weighting

            rhs = glc.constant
            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def add_max_growth(n, config):
    """
    Add maximum growth rates for different carriers.
    """

    opts = snakemake.params["sector"]["limit_max_growth"]
    # take maximum yearly difference between investment periods since historic growth is per year
    factor = n.investment_period_weightings.years.max() * opts["factor"]
    for carrier in opts["max_growth"].keys():
        max_per_period = opts["max_growth"][carrier] * factor
        logger.info(
            f"set maximum growth rate per investment period of {carrier} to {max_per_period} GW."
        )
        n.carriers.loc[carrier, "max_growth"] = max_per_period * 1e3

    for carrier in opts["max_relative_growth"].keys():
        max_r_per_period = opts["max_relative_growth"][carrier]
        logger.info(
            f"set maximum relative growth per investment period of {carrier} to {max_r_per_period}."
        )
        n.carriers.loc[carrier, "max_relative_growth"] = max_r_per_period

    return n


def add_retrofit_gas_boiler_constraint(n, snapshots):
    """
    Allow retrofitting of existing gas boilers to H2 boilers.
    """
    c = "Link"
    logger.info("Add constraint for retrofitting gas boilers to H2 boilers.")
    # existing gas boilers
    mask = n.links.carrier.str.contains("gas boiler") & ~n.links.p_nom_extendable
    gas_i = n.links[mask].index
    mask = n.links.carrier.str.contains("retrofitted H2 boiler")
    h2_i = n.links[mask].index

    n.links.loc[gas_i, "p_nom_extendable"] = True
    p_nom = n.links.loc[gas_i, "p_nom"]
    n.links.loc[gas_i, "p_nom"] = 0

    # heat profile
    cols = n.loads_t.p_set.columns[
        n.loads_t.p_set.columns.str.contains("heat")
        & ~n.loads_t.p_set.columns.str.contains("industry")
        & ~n.loads_t.p_set.columns.str.contains("agriculture")
    ]
    profile = n.loads_t.p_set[cols].div(
        n.loads_t.p_set[cols].groupby(level=0).max(), level=0
    )
    # to deal if max value is zero
    profile.fillna(0, inplace=True)
    profile.rename(columns=n.loads.bus.to_dict(), inplace=True)
    profile = profile.reindex(columns=n.links.loc[gas_i, "bus1"])
    profile.columns = gas_i

    rhs = profile.mul(p_nom)

    dispatch = n.model["Link-p"]
    active = get_activity_mask(n, c, snapshots, gas_i)
    rhs = rhs[active]
    p_gas = dispatch.sel(Link=gas_i)
    p_h2 = dispatch.sel(Link=h2_i)

    lhs = p_gas + p_h2

    n.model.add_constraints(lhs == rhs, name="gas_retrofit")


def prepare_network(
    n,
    solve_opts=None,
    config=None,
    foresight=None,
    planning_horizons=None,
    co2_sequestration_potential=None,
):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.links_t.p_max_pu,
            n.links_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if load_shedding := solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=buses_i,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if foresight == "myopic":
        add_land_use_constraint(n, planning_horizons, config)

    if foresight == "perfect":
        n = add_land_use_constraint_perfect(n)
        if snakemake.params["sector"]["limit_max_growth"]["enable"]:
            n = add_max_growth(n, config)

    if n.stores.carrier.eq("co2 sequestered").any():
        limit = co2_sequestration_potential
        add_co2_sequestration_limit(n, config, limit=limit)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"], index_col=[0, 1]
    )
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = pd.concat([gens.bus.map(n.buses.country), gens.carrier])
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )

def enforce_autarky(n, only_crossborder, level):
    n.remove(
        "GlobalConstraint",
        "lv_limit",
        )
    
    location = (
        n.buses.location
        if "location" in n.buses.columns
        else pd.Series(n.buses.index, index=n.buses.index)
    )
    def group(df, b="bus"):
        if only_crossborder:
            return df[b].map(location).map(n.buses.country).to_xarray()
        else:
            return df[b].map(location).to_xarray()
        
    lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
    link_import_carriers = [
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "gas pipeline",
        "gas pipeline new",
        #"DC",
          ]
    links_rm = (
        n.links.loc[(group(n.links, b="bus0") != group(n.links, b="bus1")).to_numpy()]  
        .loc[n.links.carrier.isin(link_import_carriers)]
        .index
    )
    links_DC = (
        n.links.loc[(group(n.links, b="bus0") != group(n.links, b="bus1")).to_numpy()]  
        .loc[n.links.carrier=="DC"]
        .index
    )
    #level = 0.999 if  level==1 else level  ##removes all gas/H2 pipelines and AC/DC lines-> post-process error
    #    n.mremove("Line", lines_rm)
    #    n.mremove("Link", links_rm)
    #    n.mremove("Link", links_DC)

    if 1==1:
        n.lines.loc[lines_rm, "s_nom_max"] *= (1-level)
        for parameter in ["s_nom", "s_nom_min", "s_nom_opt"]:
            idx = [i for i in lines_rm if n.lines.loc[i, parameter] > n.lines.loc[i, "s_nom_max"]]
            n.lines.loc[idx, parameter] = n.lines.loc[idx, "s_nom_max"]
            n.lines.loc[idx, 's_nom_extendable'] = False
            print(idx)

        n.links.loc[links_DC, "p_nom_max"] *= (1-level)
        n.links.loc[links_rm, "p_nom_max"] = 32000 * (1-level)  ##based on max installed in 2050 for gas/H2
        for parameter in ["p_nom", "p_nom_min", "p_nom_opt"]:
            idx = [i for i in links_DC if n.links.loc[i, parameter] > n.links.loc[i, "p_nom_max"]]
            n.links.loc[idx, parameter] = n.links.loc[idx, "p_nom_max"]
            n.links.loc[idx, 'p_nom_extendable'] = False

            idx_ = [i for i in links_rm if n.links.loc[i, parameter] > n.links.loc[i, "p_nom_max"]]
            n.links.loc[idx_, parameter] = n.links.loc[idx_, "p_nom_max"] 
            n.links.loc[idx_, 'p_nom_extendable'] = False

        if level == 1:
            #n.lines.loc[lines_rm, 's_nom_extendable'] = False
            #n.links.loc[links_DC, 'p_nom_extendable'] = False
            #n.links.loc[links_rm, 'p_nom_extendable'] = False
            n.mremove("Link", links_rm)
            n.mremove("Link", links_DC) 
            n.mremove("Line", lines_rm)
            #n.lines.loc[lines_rm, 's_max_pu'] = 0
            print('removing all DC/gas/H2 links')

def add_EQ_constraints(n, level, by_country, config, hourly):
    if config["foresight"] != "overnight":
        logging.warning(
            "Careful! Equity constraint is only tested for 'overnight' "
            f"foresight models, not '{config['foresight']}' foresight"
        )

    # While we need to group components by bus location in the
    # sector-coupled model, there is no "location" column in the
    # electricity-only model.
    location = (
        n.buses.location
        if "location" in n.buses.columns
        else pd.Series(n.buses.index, index=n.buses.index)
    )

    def group(df, b="bus"):
        """
        Group given dataframe by bus location or country.

        The optional argument `b` allows clustering by bus0 or bus1 for
        lines and links.
        """
        if by_country:
            return df[b].map(location).map(n.buses.country).to_xarray()
        else:
            return df[b].map(location).to_xarray()

    # Local production by generators. Note: the network may not
    # actually have all these generators (for instance some
    # conventional generators are implemented as links in the
    # sector-coupled model; heating sector might not be turned on),
    # but we list all that might be in the network.
    local_gen_carriers = list(
        set(
            config["electricity"]["extendable_carriers"]["Generator"]
            + config["electricity"]["conventional_carriers"]
            + config["electricity"]["renewable_carriers"]
            + [c for c in n.generators.carrier if "solar thermal" in c]
            + ["solar rooftop", "solar-delta", "wave","ror"] #,"solar inv","solar-hsat inv","solar-delta"]   ##added ror, alternate solar techs
        )
    )
    local_gen_i = n.generators.loc[
        n.generators.carrier.isin(local_gen_carriers)
        & (n.generators.bus.map(location) != "EU")
    ].index
    local_gen_p = (
        n.model["Generator-p"]
        .loc[:, local_gen_i]
        .groupby(group(n.generators.loc[local_gen_i]))
        .sum()
    )
    local_gen = (local_gen_p * n.snapshot_weightings.generators).sum("snapshot")

    print('gen. carriers considered for equity:', set(n.generators.carrier.loc[local_gen_i]))

    # Hydro production; the only local production from a StorageUnit.
    local_hydro_i = n.storage_units.loc[n.storage_units.carrier == "hydro"].index
    local_hydro_p = (
        n.model["StorageUnit-p_dispatch"]
        .loc[:, local_hydro_i]
        .groupby(group(n.storage_units.loc[local_hydro_i]))
        .sum()
    )
    local_hydro = (local_hydro_p * n.snapshot_weightings.stores).sum("snapshot")

    # Biomass and biogas; these are only considered locally produced
    # if spatially resolved, not if they belong to an "EU" node. They
    # are modelled as stores with initial capacity to model a finite
    # yearly supply; the difference between initial and final capacity
    # is the total local production.
    local_bio_i = n.stores.loc[
        n.stores.carrier.isin(["biogas", "solid biomass"])
        & (n.stores.bus.map(location) != "EU")
    ].index
    # Building the following linear expression only works if it's non-empty
    if len(local_bio_i) > 0:
        local_bio_first_e = n.model["Store-e"].loc[n.snapshots[0], local_bio_i]
        local_bio_last_e = n.model["Store-e"].loc[n.snapshots[-1], local_bio_i]
        local_bio_p = local_bio_first_e - local_bio_last_e
        local_bio = local_bio_p.groupby(group(n.stores.loc[local_bio_i])).sum()
    else:
        local_bio = None

    # Conventional generation in the sector-coupled model. These are
    # modelled as links in order to take the CO2 cycle into account.
    # All of these are counted as local production even if the links
    # may take their fuel from an "EU" node, except for gas and oil,
    # which are modelled endogenously and is counted under imports /
    # exports.
    ## consider gas as import when not spatially resolved with link_generate_carriers
    link_generate_carriers = [
        "residential rural gas boiler",
        "services rural gas boiler",
        "residential urban decentral gas boiler",
        "services urban decentral gas boiler",
        "urban central gas boiler",
        "urban central gas CHP",
        "urban central gas CHP CC",
        "residential rural micro gas CHP",
        "services rural micro gas CHP",
        "residential urban decentral micro gas CHP",
        "services urban decentral micro gas CHP",
        "OCGT",
        "CCGT",
    ]

    conv_carriers = config["sector"].get("conventional_generation", {})
    conv_carriers = [
        gen for gen, carrier in conv_carriers.items() if carrier not in ["gas", "oil"]
    ]
    ## link_generate_carriers are added if gas_network not present
    conv_carriers=conv_carriers #+link_generate_carriers

    if config["sector"].get("coal_cc") and not "coal" in conv_carriers:
        conv_carriers.append("coal")
    local_conv_gen_i = n.links.loc[n.links.carrier.isin(conv_carriers)].index

    if len(local_conv_gen_i) > 0:
        local_conv_gen_p = n.model["Link-p"].loc[:, local_conv_gen_i]
        # These links have efficiencies, which we multiply by since we
        # only want to count the _output_ of each conventional
        # generator as local generation for the equity balance.
        efficiencies = n.links.loc[local_conv_gen_i, "efficiency"]
        local_conv_gen_p = (
            (local_conv_gen_p * efficiencies)
            .groupby(group(n.links.loc[local_conv_gen_i], b="bus1"))
            .sum()
            .rename({"bus1": "bus"})
        )
        ## fix CHP production
        """
        local_conv_gen_chp = n.links.loc[n.links.carrier.isin([i for i in conv_carriers if 'CHP' in i])].index
        local_conv_gen_p += (
            (n.model["Link-p"].loc[:, local_conv_gen_chp] * n.links.loc[local_conv_gen_chp, "efficiency2"])
            .groupby(group(n.links.loc[local_conv_gen_chp], b="bus1"))
            .sum()
            .rename({"bus1": "bus"})
        )
        """

        local_conv_gen = (local_conv_gen_p * n.snapshot_weightings.generators).sum(
            "snapshot"
        )
    else:
        local_conv_gen = None

    #print(conv_carriers)
    #print(local_conv_gen.loc['AT'])


    # TODO: should we (in prepare_sector_network.py) model gas
    # pipeline imports from outside the EU and LNG imports separately
    # from gas extraction / production? Then we could model gas
    # extraction as locally produced energy.

    # Ambient heat for heat pumps
    heat_pump_i = n.links.filter(like="heat pump", axis="rows").index
    if len(heat_pump_i) > 0:
        # To get the ambient heat extracted, we subtract 1 from the
        # efficiency of the heat pump (where "efficiency" is really COP
        # for heat pumps).
        from_ambient = n.links_t["efficiency"].loc[:, heat_pump_i] - 1
        local_heat_from_ambient_p = n.model["Link-p"].loc[:, heat_pump_i]
        local_heat_from_ambient_ = (
            (local_heat_from_ambient_p * from_ambient)
            .groupby(group(n.links.loc[heat_pump_i], b="bus1"))
            .sum()
            .rename({"bus1": "bus"})
        )
        local_heat_from_ambient = (
            local_heat_from_ambient_ * n.snapshot_weightings.generators
        ).sum("snapshot")
    else:
        local_heat_from_ambient = None



    # Now it's time to collect imports: electricity, hydrogen & gas
    # pipeline, other gas, biomass, gas terminals & production.

    # Start with net electricity imports.
    lines_cross_region_i = n.lines.loc[
        (group(n.lines, b="bus0") != group(n.lines, b="bus1")).to_numpy()
    ].index
    # Build linear expression representing net imports (i.e. imports -
    # exports) for each bus/country.
    lines_in_s = (
        n.model["Line-s"]
        .loc[:, lines_cross_region_i]
        .groupby(group(n.lines.loc[lines_cross_region_i], b="bus1"))
        .sum()
        .rename({"bus1": "bus"})
    ) - (
        n.model["Line-s"]
        .loc[:, lines_cross_region_i]
        .groupby(group(n.lines.loc[lines_cross_region_i], b="bus0"))
        .sum()
        .rename({"bus0": "bus"})
    )
    line_imports = (lines_in_s * n.snapshot_weightings.generators).sum("snapshot")

    #print(line_imports.loc['AT'])  

    # Link net imports, representing all net energy imports of various
    # carriers that are implemented as links. We list all possible
    # link carriers that could be represented in the network; some
    # might not be present in some networks depending on the sector
    # configuration. Note that we do not count efficiencies here (e.g.
    # for oil boilers that import oil) since efficiency losses are
    # counted as "local demand".

    ##these two processes are not normally connected (triangle shape : node_h2 bus --- EU methaol -- node_shippng methanol)
    ## therefore : local + net != demand , instead : demand = imports -> local >= demand*c becomes : net-(1/c)* local <=0
    link_fuel_production = [  
        "Fischer-Tropsch",          
        "methanolisation",         
    ]

    link_import_carriers = [
        # Pipeline imports / exports
        "H2 pipeline",
        "H2 pipeline retrofitted",
        "gas pipeline",
        "gas pipeline new",
        #"CO2 pipeline",
        # Solid biomass
        "solid biomass transport",  ##not in model?
        # DC electricity
        "DC",
        # Oil (imports / exports between spatial nodes and "EU" node)
        #"Fischer-Tropsch",         ## local production        
        "biomass to liquid",      ##not in model?
        ##"residential rural oil boiler",
        ##"services rural oil boiler",
        ##"residential urban decentral oil boiler",
        ##"services urban decentral oil boiler",
        "oil",  # Oil powerplant (from `prepare_sector_network.add_generation`)
        # Gas (imports / exports between spatial nodes and "EU" node,
        # only cross-region if gas is not spatially resolved)
        #"Sabatier",      ##   in country with gas/H2 network          
        "helmeth",   ##not in model?
        ##"SMR CC",                ##in country with gas network
        ##"SMR",                ##in country with gas network
        ##"biogas to gas",       ##in country with gas network
        "BioSNG",                   ##not in model?
        #"methanolisation",        ## local production   
        ##"residential rural gas boiler",
        ##"services rural gas boiler",
        ##"residential urban decentral gas boiler",
        ##"services urban decentral gas boiler",
        ##"urban central gas boiler",
        ##"urban central gas CHP",
        ##"urban central gas CHP CC",
        ##"residential rural micro gas CHP",
        ##"services rural micro gas CHP",
        ##"residential urban decentral micro gas CHP",
        ##"services urban decentral micro gas CHP",
        "allam",
        ##"OCGT",
        ##"CCGT",
        ## Extra imports 
        "shipping oil",
        "agriculture machinery oil",
        "land transport oil",
        "shipping methanol",
        "kerosene for aviation",
        "naphtha for industry",
    ]
    links_cross_region_i = (
        n.links.loc[(group(n.links, b="bus0") != group(n.links, b="bus1")).to_numpy()]  ##all EU->EU links (in case of not being spatially resolved) are disregarded
        .loc[n.links.carrier.isin(link_import_carriers)]
        .index
    )

    links_fuel_region_i = (
        n.links.loc[(group(n.links, b="bus0") != group(n.links, b="bus1")).to_numpy()]  ##all EU->EU links (in case of not being spatially resolved) are disregarded
        .loc[n.links.carrier.isin(link_fuel_production)]
        .index
    )
    #print(links_cross_region_i)

    # Build linear expression representing net imports (i.e. imports -
    # exports) for each bus/country.
    links_in_p = (
        n.model["Link-p"]
        .loc[:, links_cross_region_i]
        .groupby(group(n.links.loc[links_cross_region_i], b="bus1"))
        .sum()
        .rename({"bus1": "bus"})
    ) - (
        n.model["Link-p"]
        .loc[:, links_cross_region_i]
        .groupby(group(n.links.loc[links_cross_region_i], b="bus0"))
        .sum()
        .rename({"bus0": "bus"})
    )

    link_imports = (links_in_p * n.snapshot_weightings.generators).sum("snapshot")
    print(link_imports.loc['AL'])

    links_in_p_fuel = (
        (n.model["Link-p"]
        .loc[:, links_fuel_region_i] * n.links.loc[links_fuel_region_i, "efficiency"])
        .groupby(group(n.links.loc[links_fuel_region_i], b="bus0"))
        .sum()
        .rename({"bus0": "bus"})
    )

    fuel_production = (links_in_p_fuel * n.snapshot_weightings.generators).sum("snapshot")

    # Gas imports by pipeline from outside of Europe, LNG terminal or
    # local gas production (all modelled as a single generator in each country) :disadvatage for countries with gas like Norway
    # This  does not work without gas_network as they can't be summed by country ('EU gas'). 
    gas_import_i = n.generators.loc[n.generators.carrier == "gas"].index  
    if len(gas_import_i) > 0:
        gas_import_p = (
            n.model["Generator-p"]
            .loc[:, gas_import_i]
            .groupby(group(n.generators.loc[gas_import_i]))
            .sum()
        )
        gas_imports = (gas_import_p * n.snapshot_weightings.generators).sum("snapshot")
    else:
        gas_imports = None

    local_factor = 1 - 1 / level

    #print(imported_energy.loc['AL'])
    #print(local_energy.loc['AL'])

    if not hourly: 
        # Total locally produced energy
        local_energy = sum( e for e in [
            local_gen,
            local_hydro,
            local_bio,
            local_conv_gen,
            local_heat_from_ambient,
           ]
           if e is not None
        )

        imported_energy = sum(
            i for i in [line_imports, 
                        link_imports,
                         gas_imports
                    ] 
             if i is not None   
        )
        n.model.add_constraints(
          local_factor * local_energy + imported_energy - (1 / level) * fuel_production  <= 0, name="equity_min"
        )

    else: 
        print('adding hourly equity constraint...')
        #print(gas_import_p)

        for i in range(len(n.snapshots)): 
            
      
            if len(local_bio_i) > 0:
                local_bio_p = n.model["Store-e"].loc[n.snapshots[i], local_bio_i] - n.model["Store-e"].loc[n.snapshots[i-1], local_bio_i]
                local_bio = local_bio_p.groupby(group(n.stores.loc[local_bio_i])).sum()
            else:
                local_bio = None


            local_energy = sum(
                e for e in [
                    local_gen_p.loc[n.snapshots[i],:],
                    local_hydro_p.loc[n.snapshots[i],:],
                    local_bio,
                    local_conv_gen_p .loc[n.snapshots[i],:],
                    local_heat_from_ambient_.loc[n.snapshots[i],:] ,
                    ]
                if e is not None
            ) 
            imported_energy = sum(
                m for m in [
                    lines_in_s.loc[:,n.snapshots[i]], 
                    links_in_p.loc[n.snapshots[i],:],
                    gas_import_p.loc[n.snapshots[i],:]
                            ] 
                if m is not None   
            )    
                
            fuel_production = links_in_p_fuel.loc[n.snapshots[i],:]

            if i==1:
                print(local_energy.loc['AL'])
                print(imported_energy.loc['AL'])
                print(fuel_production.loc['AL'])

            n.model.add_constraints(
                local_factor * local_energy + imported_energy - (1 / level) * fuel_production  <= 0, name="equity_min "+str(i)
                   )


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24H]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24H]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    conventional_carriers = config["electricity"]["conventional_carriers"]  # noqa: F841
    ext_gens_i = n.generators.query(
        "carrier in @conventional_carriers & p_nom_extendable"
    ).index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conventional_carriers"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    summed_reserve = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = summed_reserve + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum(
            "Generator"
        )

    # Total demand per t
    demand = get_as_dense(n, "Load", "p_set").sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    # additional constraint that capacity is not exceeded
    gen_i = n.generators.index
    ext_i = n.generators.query("p_nom_extendable").index
    fix_i = n.generators.query("not p_nom_extendable").index

    dispatch = n.model["Generator-p"]
    reserve = n.model["Generator-r"]

    capacity_variable = n.model["Generator-p_nom"].rename(
        {"Generator-ext": "Generator"}
    )
    capacity_fixed = n.generators.p_nom[fix_i]

    p_max_pu = get_as_dense(n, "Generator", "p_max_pu")

    lhs = dispatch + reserve - capacity_variable * p_max_pu[ext_i]

    rhs = (p_max_pu[fix_i] * capacity_fixed).reindex(columns=gen_i, fill_value=0)

    n.model.add_constraints(lhs <= rhs, name="Generator-p-reserve-upper")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_lossy_bidirectional_link_constraints(n):
    if not n.links.p_nom_extendable.any() or "reversed" not in n.links.columns:
        return

    n.links["reversed"] = n.links.reversed.fillna(0).astype(bool)
    carriers = n.links.loc[n.links.reversed, "carrier"].unique()  # noqa: F841

    forward_i = n.links.query(
        "carrier in @carriers and ~reversed and p_nom_extendable"
    ).index

    def get_backward_i(forward_i):
        return pd.Index(
            [
                re.sub(r"-(\d{4})$", r"-reversed-\1", s)
                if re.search(r"-\d{4}$", s)
                else s + "-reversed"
                for s in forward_i
            ]
        )

    backward_i = get_backward_i(forward_i)

    lhs = n.model["Link-p_nom"].loc[backward_i]
    rhs = n.model["Link-p_nom"].loc[forward_i]

    n.model.add_constraints(lhs == rhs, name="Link-bidirectional_sync")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    if "reversed" not in n.links.columns:
        n.links["reversed"] = False
    gas_pipes_i = n.links.query(
        "carrier == 'gas pipeline' and p_nom_extendable and ~reversed"
    ).index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable and ~reversed"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")


def add_co2_atmosphere_constraint(n, snapshots):
    glcs = n.global_constraints[n.global_constraints.type == "co2_atmosphere"]

    if glcs.empty:
        return
    for name, glc in glcs.iterrows():
        carattr = glc.carrier_attribute
        emissions = n.carriers.query(f"{carattr} != 0")[carattr]

        if emissions.empty:
            continue

        # stores
        n.stores["carrier"] = n.stores.bus.map(n.buses.carrier)
        stores = n.stores.query("carrier in @emissions.index and not e_cyclic")
        if not stores.empty:
            last_i = snapshots[-1]
            lhs = n.model["Store-e"].loc[last_i, stores.index]
            rhs = glc.constant

            n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    constraints = config["solving"].get("constraints", {})
    if (
        "BAU" in opts or constraints.get("BAU", False)
    ) and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if (
        "SAFE" in opts or constraints.get("SAFE", False)
    ) and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if (
        "CCL" in opts or constraints.get("CCL", False)
    ) and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)

    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)

    #EQ_config = constraints.get("EQ", False)
    #EQ_wildcard = get_opt(opts, r"^EQ+[0-9]*\.?[0-9]+(c|)")
    #EQ_o = EQ_wildcard or EQ_config
    #if EQ_o:
    #    add_EQ_constraints(n, EQ_o.replace("EQ", ""))
    sector_opts_ = snakemake.wildcards.sector_opts.split("-")
    investment_year = int(snakemake.wildcards.planning_horizons[-4:])
    add_wind_potential_constraints(n, config)

    for o in sector_opts_:
        if 'newsolar' in o:
           print('adding solar land use constraints...')
           add_solar_potential_constraints(n, config) ##config   

        if "EQ" in o:
            float_regex = "[0-9]*\.?[0-9]+"
            #level = float(re.findall(float_regex, o)[0])
            level = get(snakemake.params.equity, investment_year)

            #EQ_regex = "EQ(0\.[0-9]+)(c?)"  # Ex.: EQ0.75c
            #m = re.search(EQ_regex, o)
            if level is not None:   # m is not None
                #level = float(m.group(1))
                #by_country = True if m.group(2) == "c" else False
                by_country = True if "c" in o else False
                hourly= True if "h" in o else False
                print('EQ ratio is', level , 'by country is', by_country, 'hourly is', hourly)
                add_EQ_constraints(n, level, by_country, config, hourly)
            else:
                logging.warning(f"Invalid EQ option: {o}")

        if "AUT" in o:     
            level = get(snakemake.params.autarky, investment_year)
            only_crossborder = True if "c" in o else False  ##by country
            print('Autarky ratio is', level , 'by country is', only_crossborder,)
            enforce_autarky(n, only_crossborder, level)   
     

    add_battery_constraints(n)
    add_lossy_bidirectional_link_constraints(n)
    add_pipe_retrofit_constraint(n)
    if n._multi_invest:
        add_carbon_constraint(n, snapshots)
        add_carbon_budget_constraint(n, snapshots)
        add_retrofit_gas_boiler_constraint(n, snapshots)
    else:
        add_co2_atmosphere_constraint(n, snapshots)

    if snakemake.params.custom_extra_functionality:
        source_path = snakemake.params.custom_extra_functionality
        assert os.path.exists(source_path), f"{source_path} does not exist"
        sys.path.append(os.path.dirname(source_path))
        module_name = os.path.splitext(os.path.basename(source_path))[0]
        module = importlib.import_module(module_name)
        custom_extra_functionality = getattr(module, module_name)
        custom_extra_functionality(n, snapshots, snakemake)


def solve_network(n, config, solving, opts="", **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["multi_investment_periods"] = config["foresight"] == "perfect"
    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality
    kwargs["transmission_losses"] = cf_solving.get("transmission_losses", False)
    kwargs["linearized_unit_commitment"] = cf_solving.get(
        "linearized_unit_commitment", False
    )
    kwargs["assign_all_duals"] = cf_solving.get("assign_all_duals", False)
    kwargs["io_api"] = cf_solving.get("io_api", None)

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)

    rolling_horizon = cf_solving.pop("rolling_horizon", False)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    tmpdir_scratch = '/scratch/' + os.environ['SLURM_JOB_ID']

    if tmpdir_scratch is not None:
        Path(tmpdir_scratch).mkdir(parents=True, exist_ok=True)

    if rolling_horizon:
        kwargs["horizon"] = cf_solving.get("horizon", 365)
        kwargs["overlap"] = cf_solving.get("overlap", 0)
        n.optimize.optimize_with_rolling_horizon(model_kwargs={"solver_dir":tmpdir_scratch},  ##
            **kwargs)
        status, condition = "", ""
    elif skip_iterations:
        status, condition = n.optimize(model_kwargs={"solver_dir":tmpdir_scratch},  ##
            **kwargs)
    else:
        kwargs["track_iterations"] = (cf_solving.get("track_iterations", False),)
        kwargs["min_iterations"] = (cf_solving.get("min_iterations", 4),)
        kwargs["max_iterations"] = (cf_solving.get("max_iterations", 6),)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(model_kwargs={"solver_dir":tmpdir_scratch},  ##
            **kwargs
        )

    if status != "ok" and not rolling_horizon:
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            configfiles="../config/test/config.perfect.yaml",
            simpl="",
            opts="",
            clusters="37",
            ll="v1.0",
            sector_opts="CO2L0-1H-T-H-B-I-A-dist1",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config, snakemake.wildcards.sector_opts
        )

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    n = prepare_network(
        n,
        solve_opts,
        config=snakemake.config,
        foresight=snakemake.params.foresight,
        planning_horizons=snakemake.params.planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
    )

    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=30.0
    ) as mem:
        n = solve_network(
            n,
            config=snakemake.config,
            solving=snakemake.params.solving,
            opts=opts,
            log_fn=snakemake.log.solver,
        )


    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])

    # write dual values into a CSV file
    dual_values_file = "results/%s/dual_values_s%s_%s_l%s_%s_%s_%s.csv" % (snakemake.config["run"]["name"], snakemake.wildcards.simpl, snakemake.wildcards.clusters, snakemake.wildcards.ll, snakemake.wildcards.opts, snakemake.wildcards.sector_opts, snakemake.wildcards.planning_horizons)
    logger.info("Write dual values to file '%s'" % dual_values_file)
    with open(dual_values_file, "w") as handle:
            handle.write("node,value\n")
            suffix = "equity_min"
            for key in n.model.dual.keys():
                    if key.endswith(suffix):
                        value = n.model.dual[key].values
                        print(key, value)
                        #handle.write("%s,%f\n" % (key[:-len(suffix)], value))

    logger.info(f"Maximum memory usage: {mem.mem_usage}")
