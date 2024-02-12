num_iterations=150

for ((i=1; i<=$num_iterations; i++))
do
	echo "bash: running iteration $i / $num_iterations"
	taskset -c 0 python injection_recovery.py
done
