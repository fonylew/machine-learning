-- knapsack 5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/ga__Knapsack_5__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__Knapsack_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_5/ga__Knapsack_5__curves_df.csv'
where "Population Size" = 150  and "Mutation Rate" = 0.6 
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__Knapsack_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- knapsack 10
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/ga__Knapsack_10__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__Knapsack_10__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/Knapsack_10/ga__Knapsack_10__curves_df.csv'
where "Population Size" = 200 and "Mutation Rate" = 0.5
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__Knapsack_10__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 20
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/ga__continuous_peaks__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__continuous_peaks__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks/ga__continuous_peaks__curves_df.csv'
where "Population Size" = 150 and "Mutation Rate" = 0.6
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__continuous_peaks__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- c. peaks 30
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/ga__continuous_peaks30__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__continuous_peaks30__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/continuous_peaks30/ga__continuous_peaks30__curves_df.csv'
where "Population Size" = 150 and "Mutation Rate" = 0.4
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__continuous_peaks30__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_5
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/ga__TSP_5__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__TSP_5__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_5/ga__TSP_5__curves_df.csv'
where "Population Size" = 150 and "Mutation Rate" = 0.4
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__TSP_5__curves.csv' WITH (HEADER 1, DELIMITER ',');

-- TSP_22
copy(
SELECT max(Fitness) as max_fitness, max(FEvals) as max_evals, max(Iteration) as iter, max(Time) as max_time, "Population Size", "Mutation Rate"
FROM '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/ga__TSP_22__curves_df.csv'
group by 5,6
order by max_fitness desc, max_evals
) TO '/Users/fony/machine-learning/randomized_optimization/selected_summary/ga__TSP_22__summary.csv' WITH (HEADER 1, DELIMITER ',');

copy(
select * from '/Users/fony/machine-learning/randomized_optimization/results/TSP_22/ga__TSP_22__curves_df.csv'
where "Population Size" = 150 and "Mutation Rate" = 0.4
order by Iteration
) TO '/Users/fony/machine-learning/randomized_optimization/selected_results/ga__TSP_22__curves.csv' WITH (HEADER 1, DELIMITER ',');
