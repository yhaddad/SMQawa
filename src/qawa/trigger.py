def add_trigger_sf(weight, event: processor.LazyDataFrame):
    # Define all the categories
    mask_mm = (event.lep_category == 3) | (event.lep_category == 5) | (event.lep_category == 7)
    mask_ee = (event.lep_category == 1) | (event.lep_category == 4) | (event.lep_category == 6)
    mask_me = (~mask_mm & ~mask_ee) & (event.leading_lep_flavor == 1)
    mask_em = (~mask_mm & ~mask_ee) & (event.leading_lep_flavor == 0)
    mask_BB = (event.leading_lep_eta <= 1.5) & (event.trailing_lep_eta <= 1.5) # 0
    mask_EB = (event.leading_lep_eta >= 1.5) & (event.trailing_lep_eta <= 1.5) # 1
    mask_BE = (event.leading_lep_eta <= 1.5) & (event.trailing_lep_eta >= 1.5) # 2
    mask_EE = (event.leading_lep_eta >= 1.5) & (event.trailing_lep_eta >= 1.5) # 3
    
    # these are the pt_bins used for the scale factor 
    lept_pt_bins = [20, 25, 30, 35, 40, 50, 60, 100000]
    lep_1_bin = np.digitize(event.leading_lep_pt.to_numpy() , lept_pt_bins) - 1
    lep_2_bin = np.digitize(event.trailing_lep_pt.to_numpy(), lept_pt_bins) - 1
    trigg_bin = np.select([
        (mask_ee & mask_BB).to_numpy(),
        (mask_ee & mask_BE).to_numpy(),
        (mask_ee & mask_EB).to_numpy(),
        (mask_ee & mask_EE).to_numpy(),

        (mask_em & mask_BB).to_numpy(),
        (mask_em & mask_BE).to_numpy(),
        (mask_em & mask_EB).to_numpy(),
        (mask_em & mask_EE).to_numpy(),

        (mask_me & mask_BB).to_numpy(),
        (mask_me & mask_BE).to_numpy(),
        (mask_me & mask_EB).to_numpy(),
        (mask_me & mask_EE).to_numpy(),

        (mask_mm & mask_BB).to_numpy(),
        (mask_mm & mask_BE).to_numpy(),
        (mask_mm & mask_EB).to_numpy(),
        (mask_mm & mask_EE).to_numpy()
    ], np.arange(0,16), 16)
    
    indices = np.column_stack([lep_1_bin, lep_2_bin,trigg_bin])
    center_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,0]
    errors_value = self.trig_sf_map[lep_1_bin,lep_2_bin,trigg_bin,1]
    
    event['TriggerSFWeight'] = center_value
    event['TriggerSFWeightUp'] = center_value + errors_value
    event['TriggerSFWeightDown'] = center_value - errors_value 
    
    return event
