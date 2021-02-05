from collections import Counter 
import math

def a(data):
    data = data.lower()
    count = Counter(data)  # Count of each letter
    
    factorials = [math.factorial(x) for x in list(count.values())]
    factorial_prod = 1# int(round(math.exp(sum(map(math.log, factorials))))) #Find the product of factorials
    for i in factorials:
        factorial_prod = factorial_prod * i
    
    factorial_top = int((math.factorial(int(len(data)))))
    print("factorials", factorials)
    print(count)
    print(factorial_top, factorial_prod)
    return  factorial_top // factorial_prod
    