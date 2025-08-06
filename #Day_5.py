# Day_5
# Write a python program to input ‘n’ employee salaries and find the minimum and maximum salary amongst ‘n’ employees
n = int(input("Enter the number of employees: "))
salaries = []

for i in range(n):
    salary = float(input(f"Enter the salary of employee {i+1}: "))
    salaries.append(salary)

salaries = tuple(salaries)

min_salary = min(salaries)
max_salary = max(salaries)

print(f"Salaries (tuple): {salaries}")
print(f"Minimum salary: ₹{min_salary}")
print(f"Maximum salary: ₹{max_salary}")
