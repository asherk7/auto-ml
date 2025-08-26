"""
Examples package for the AutoML system.
Contains example scripts demonstrating how to use the AutoML system
for various computer vision and tabular machine learning tasks.
"""

# Import example functions for easy access
try:
    from .cifar10_example import main as run_cifar10_example
except ImportError:
    run_cifar10_example = None

try:
    from .iris_example import main as run_iris_example
except ImportError:
    run_iris_example = None

__all__ = [
    'run_cifar10_example',
    'run_iris_example'
]

def list_examples():
    """List available examples"""
    examples = []

    if run_cifar10_example:
        examples.append({
            'name': 'CIFAR-10 Classification',
            'function': 'run_cifar10_example',
            'description': 'Train a ResNet-18 model on CIFAR-10 dataset for image classification',
            'file': 'cifar10_example.py'
        })

    if run_iris_example:
        examples.append({
            'name': 'Iris Classification',
            'function': 'run_iris_example',
            'description': 'Train multiple tabular models on Iris dataset for classification',
            'file': 'iris_example.py'
        })

    return examples

def run_all_examples():
    """Run all available examples"""
    results = {}

    if run_cifar10_example:
        print("Running CIFAR-10 example...")
        try:
            results['cifar10'] = run_cifar10_example()
            print("✓ CIFAR-10 example completed successfully")
        except Exception as e:
            print(f"✗ CIFAR-10 example failed: {e}")
            results['cifar10'] = {'error': str(e)}

    if run_iris_example:
        print("\nRunning Iris example...")
        try:
            results['iris'] = run_iris_example()
            print("✓ Iris example completed successfully")
        except Exception as e:
            print(f"✗ Iris example failed: {e}")
            results['iris'] = {'error': str(e)}

    return results

# Example usage information
USAGE = """
AutoML Examples Usage:

1. Run individual examples:
   python examples/cifar10_example.py
   python examples/iris_example.py

2. Use from Python:
   from examples import run_cifar10_example, run_iris_example

   # Run CIFAR-10 example
   results = run_cifar10_example()

   # Run Iris example
   results = run_iris_example()

   # Run all examples
   from examples import run_all_examples
   all_results = run_all_examples()

3. List available examples:
   from examples import list_examples
   examples = list_examples()
   for example in examples:
       print(f"- {example['name']}: {example['description']}")
"""
