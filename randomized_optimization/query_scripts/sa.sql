-- knapsack 5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/sa__Knapsack_5__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__Knapsack_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/sa__Knapsack_5__curves_df.csv'
where Temperature = 2500
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__Knapsack_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- knapsack 10
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/sa__Knapsack_10__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__Knapsack_10__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/sa__Knapsack_10__curves_df.csv'
where Temperature = 10000
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__Knapsack_10__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 20
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/sa__continuous_peaks__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__continuous_peaks__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/sa__continuous_peaks__curves_df.csv'
where Temperature = 1
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__continuous_peaks__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 30
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/sa__continuous_peaks30__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__continuous_peaks30__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/sa__continuous_peaks30__curves_df.csv'
where Temperature = 1
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__continuous_peaks30__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/sa__TSP_5__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__TSP_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/sa__TSP_5__curves_df.csv'
where Temperature = 250
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__TSP_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_22
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, Temperature, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/sa__TSP_22__curves_df.csv'
group by 5,6
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/sa__TSP_22__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/sa__TSP_22__curves_df.csv'
where Temperature = 5000
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/sa__TSP_22__curves.csv' WITH (HEADER 1, DELIMITER ',');
