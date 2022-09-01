echo "Starting... GERMAN NULL"
python GERMAN_comparison_10_executions.py | tee -a GERMAN_results_null.txt
echo "Starting... SHOPPING NULL"
python SHOPPING_comparison_10_executions.py | tee -a SHOPPING_results_null.txt
echo "Starting... GERMAN RELU"
python GERMAN_comparison_10_executions_relu.py | tee -a GERMAN_results_relu.txt
echo "Starting... SHOPPING RELU"
python SHOPPING_comparison_10_executions_relu.py | tee -a SHOPPING_results_relu.txt



echo "Starting... MNIST NULL"
python MNIST_comparison_10_executions.py | tee -a MNIST_results_null.txt
echo "Starting... MNIST RELU"
python MNIST_comparison_10_executions_relu.py | tee -a MNIST_results_relu.txt
echo "Starting... CIFAR10 NULL"
python CIFAR10_comparison_10_executions.py | tee -a CIFAR10_results_null.txt
echo "Starting... CIFAR10 RELU"
python CIFAR10_comparison_10_executions_relu.py | tee -a CIFAR10_results_relu.txt
