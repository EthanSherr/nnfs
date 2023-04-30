names = ['ethan', 'anna', 'may', 'fred']
ages = [33, 36, 28, 35]

for name, age in zip(names, ages):
    print('name', name, 'age', age)

what = zip(names, ages)
print('what', what)