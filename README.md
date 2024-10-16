# Strategic deployment of solar photovoltaics for achieving self-sufficiency in Europe throughout the energy transition"

This repository contains the scripts used to generate network files and the notebooks to reproduce figures in the main text and supplementary of the paper.

## Abstract

Transition pathways for Europe to achieve carbon neutrality emphasize the need for a massive deployment of solar and wind energy.
Global cost optimization would lead to installing most of the renewable capacity in a few resource-rich countries, but policy decisions could prioritize other factors.
We investigate the effect of energy independence on Europe’s energy system design. We show that self-sufficiency constraints lead to a more equitable distribution of costs
and installed capacities across Europe. However, countries that typically depend on energy imports face cost increases of up to 150\% to achieve complete self-sufficiency.
Self-sufficiency particularly favours solar photovoltaic (PV) energy, and with declining PV module prices, alternative configurations like inverter dimensioning and horizontal
tracking are beneficial enough to be part of the optimal solution for many countries. Moreover, we found that very large solar and wind annual installation rates are required,
but they seem feasible considering recent historical trends.

## Repository Structure

- `notebooks` contains the Jupyter notebooks used for the evaluation of results.
- `scripts` and 'config files' can be used to generate the network files using pypsa-eur\href{https://github.com/PyPSA/pypsa-eur}.
Attention:
1. The scripts were used with pypsa-eur v0.9.0 and atlite v0.2.12. Adjustments to the code may be needed if later versions of pypsa-eur and atlite are used.
2. The notebooks is customised for networks of this study, so care should be taken
when using it to produce similiar figures for PyPSA network files. 

%## Usage

%The notebooks use pre-solved networks (main scenario files avialble at zenodo:https://zenodo.org/records/11277299) to produce the figures from the paper.
%PyPSA-Eur-Sec v0.6.0 is used to produce the networks.
 
