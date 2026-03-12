# Presentation Summary

- Run mode: full
- Main runs: 18000
- Main summary rows: 450
- Main agreement rate: 0.4201
- Main strategies observed: BoulwareTBNegotiator, ConcederTBNegotiator, GeniusBOABaselineNegotiator, LinearTBNegotiator, SmartAspirationNegotiator
- Best ordered pairing by welfare: ConcederTBNegotiator vs ConcederTBNegotiator (mean=0.7961)
- Best ordered pairing by Pareto distance: ConcederTBNegotiator vs ConcederTBNegotiator (mean=0.2736)
- Multilateral runs: 15000
- Multilateral agreement rate: 0.3979
- appendix_scores.csv written with aggregated tournament metrics; `advantage` is directly produced by NegMAS.

## Top Strategies (Role-Aggregated)

```text
                   strategy  own_utility_mean  agreement_rate
  SmartAspirationNegotiator          0.275291        0.398056
       BoulwareTBNegotiator          0.232422        0.341389
         LinearTBNegotiator          0.231506        0.460278
GeniusBOABaselineNegotiator          0.227803        0.340139
       ConcederTBNegotiator          0.168957        0.560694
```

## Ordered Pairing Snapshot

```text
                                            ordered_pair  agreement_rate  welfare_mean  pareto_distance_mean  rounds_agreement_mean  count
            ConcederTBNegotiator vs ConcederTBNegotiator        0.815278      0.796076              0.273587               2.567291    720
              LinearTBNegotiator vs ConcederTBNegotiator        0.681944      0.756787              0.284592               6.627291    720
              ConcederTBNegotiator vs LinearTBNegotiator        0.668056      0.705375              0.287955               7.066528    720
       SmartAspirationNegotiator vs ConcederTBNegotiator        0.551389      0.579026              0.373584              12.964736    720
                LinearTBNegotiator vs LinearTBNegotiator        0.500000      0.544607              0.438586              11.083333    720
         SmartAspirationNegotiator vs LinearTBNegotiator        0.476389      0.539045              0.431133              15.256560    720
  SmartAspirationNegotiator vs SmartAspirationNegotiator        0.401389      0.482048              0.481163              17.363322    720
       SmartAspirationNegotiator vs BoulwareTBNegotiator        0.400000      0.475913              0.485207              17.423611    720
SmartAspirationNegotiator vs GeniusBOABaselineNegotiator        0.398611      0.470388              0.488672              17.477352    720
     GeniusBOABaselineNegotiator vs ConcederTBNegotiator        0.429167      0.468886              0.461107              12.019417    720
```

## Result Layout

- `main/`: raw and aggregated bilateral outputs
- `multilateral/`: raw and aggregated multilateral outputs
- `appendix/`: tournament outputs
- `figures/stats/`: bootstrap/effect-size figures
- `figures/analysis/`: presentation plots
- `logs/`: full sequential run logs