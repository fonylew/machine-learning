-- knapsack 5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/rhc__Knapsack_5__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__Knapsack_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/rhc__Knapsack_5__curves_df.csv'
where Restarts = 25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__Knapsack_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- knapsack 10
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/rhc__Knapsack_10__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__Knapsack_10__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/rhc__Knapsack_10__curves_df.csv'
where Restarts = 75
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__Knapsack_10__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 20
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/rhc__continuous_peaks__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__continuous_peaks__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/rhc__continuous_peaks__curves_df.csv'
where Restarts = 25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__continuous_peaks__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 30
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/rhc__continuous_peaks30__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__continuous_peaks30__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/rhc__continuous_peaks30__curves_df.csv'
where Restarts = 25
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__continuous_peaks30__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/rhc__TSP_5__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__TSP_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/rhc__TSP_5__curves_df.csv'
where Restarts = 75
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__TSP_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_22
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, max(current_restart) as max_current_restart, Restarts, max_iters
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/rhc__TSP_22__curves_df.csv'
group by 6,7
order by max_fitness desc, iter
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/rhc__TSP_22__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/rhc__TSP_22__curves_df.csv'
where Restarts = 100
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/rhc__TSP_22__curves.csv' WITH (HEADER 1, DELIMITER ',');
