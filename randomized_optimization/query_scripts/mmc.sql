-- knapsack 5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/mimic__Knapsack_5__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__Knapsack_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/mimic__Knapsack_5__curves_df.csv'
where "Population Size" = 150  and "Keep Percent" = 0.25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__Knapsack_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- knapsack 10
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/mimic__Knapsack_10__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__Knapsack_10__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/mimic__Knapsack_10__curves_df.csv'
where "Population Size" = 150 and "Keep Percent" = 0.25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__Knapsack_10__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 20
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/mimic__continuous_peaks__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__continuous_peaks__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/mimic__continuous_peaks__curves_df.csv'
where "Population Size" = 150 and "Keep Percent" = 0.75
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__continuous_peaks__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 30
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/mimic__continuous_peaks30__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__continuous_peaks30__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/mimic__continuous_peaks30__curves_df.csv'
where "Population Size" = 300 and "Keep Percent" = 0.25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__continuous_peaks30__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/mimic__TSP_5__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__TSP_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/mimic__TSP_5__curves_df.csv'
where "Population Size" = 150 and "Keep Percent" = 0.25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__TSP_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_22
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Keep Percent"
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/mimic__TSP_22__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/mimic__TSP_22__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/mimic__TSP_22__curves_df.csv'
where "Population Size" = 150 and "Keep Percent" = 0.25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/mimic__TSP_22__curves.csv' WITH (HEADER 1, DELIMITER ',');
