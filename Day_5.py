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



# Write a program to input ‘n’ employee numbers and names and store them in a dictionary. Display dictionaries employee’s names(key) and numbers(values) together.
def e_data(n):
    employee = {}
    for i in range(n):
        name = input(f"Enter the name of employee {i+1}: ")
        number = input(f"Enter the employee number of {name}: ")
        employee[name] = number
    return employee

def display_employees(employee):
    for name, number in employee.items():
        print(f"Name: {name}, Employee Number: {number}")


n = int(input("Enter the number of employees: "))
employees = e_data(n)
display_employees(employees)
