Script to produce capital cost for solar technologies from DEA based on Supplementary equation(2) of the paper: 

    
    Modify solar-utility cost based on inverter sizing 
    price (€/MW_dc) = DC_components_price (€/MW_dc) + (AC_components price(€/MW_ac)/sizing_factor) (€/MW_dc)
    price (€/MW_ac) = price (€/MW_dc) * sizing_factor
    for sizing_factor of 1 (DEA default is 1.25): price = 0.25+0.07/1 = 0.32 €/MW_dc = 0.32 €/MW_ac
    if (tech in ["solar-utility","solar-utility single-axis tracking","solar-rooftop residential","solar-rooftop commercial",]) : 
    ...
