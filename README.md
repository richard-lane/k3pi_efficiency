# k3pi_efficiency
Efficiency Reweighting for LHCb D0 -> K3pi analysis

## Notes
Need to properly write this out later TODO

 - flipped momenta such that RS is Dbar0 -> K+3pi and WS is D0 -> K+3pi; will need to do something to make this
   consistent with the data as well
 - Momentum ordered based on M(Kpi) for the two equivalent pions
 - Reweight AG->MC times to avoid large weights, then keep those weights for phsp weighting, then final weights
   are phsp weights * 1 / time weights
