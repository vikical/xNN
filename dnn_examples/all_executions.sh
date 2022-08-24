echo "Starting... CIFAR10"
python CIFAR10_comparison_10_executions.py | tee -a CIFAR10_results.txt
echo "Starting... MNIST"
python MNIST_comparison_10_executions.py | tee -a MNIST_results.txt
echo "Starting... GERMAN"
python GERMAN_comparison_10_executions.py | tee -a GERMAN_results.txt
echo "Starting... SHOPPING"
python SHOPPING_comparison_10_executions.py | tee -a SHOPPING_results.txt
