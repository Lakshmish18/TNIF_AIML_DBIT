"""Build a simple ATM simulation. User must:
Enter correct PIN (e.g., 1234)

Can perform withdraw, deposit, or check balance

Only 3 attempts allowed for PIN

Simple ATM Simulation"""


CORRECT_PIN = "1234"
balance = 10000  
attempts = 3

while attempts > 0:
    pin = input("Enter your pin : ")
    if pin == CORRECT_PIN:
        print("pin accepted. Welcome!\n")
        
        while True:
            print("\nATM")
            print("1. Check balance")
            print("2. deposit")
            print("3. withdraw")
            print("4. exit")
            
            choice = input("Choose an option (1-4): ")

            if choice == '1':
                print(f"Your current balance is Rs {balance}")
            
            elif choice == '2':
                amount = float(input("Enter amount to deposit: Rs"))
                if amount > 0:
                    balance += amount
                    print(f"RS {amount} deposited successfully.")
                else:
                    print("Invalid deposit amount.")
            
            elif choice == '3':
                amount = float(input("Enter amount to withdraw: Rs"))
                if 0 < amount <= balance:
                    balance -= amount
                    print(f"Rs {amount} withdrawn successfully.")
                else:
                    print("Insufficient balance or invalid amount.")
            
            elif choice == '4':
                print("Thank you for using the ATM. Goodbye!")
                break
            
            else:
                print("Invalid choice. Please try again.")

        break
    else:
        attempts -= 1
        print(f"Incorrect PIN. Attempts left: {attempts}")
        if attempts == 0:
            print("Too many incorrect attempts. Card blocked.")

sum=0
while True:
    n=int(input("enter the number:"))
    if n>300:
        print("Number is greater than 300. Exiting the loop.")
        break
    else:
        sum+=n
        if sum>300:
            print("Sum exceeded 300. Exiting the loop.")
            break
print(f"Total sum of numbers entered: {sum}")


