#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from itertools import combinations, chain
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import time

print("Here are databases you can explore....")
print("1) Stop & Shop")
print("2) Walgreens")
print("3) Kroger")
print("4) Walmart")
print("5) CVS")
      

while True:
    
    try:
        database = int(input("Pick a database by number: "))
        if 0 < database <= 5 :
            if database == 1:
                print("Stop & Shop Chosen!")
            if database == 2:
                print("Walgreens Chosen!")
            if database == 3:
                print("Kroger Chosen!")
            if database == 4:
                print("Walmart Chosen!")
            if database == 5:
                print("CVS Chosen!")
            break
        else:
            print("Error: Database number must be between 1 and 5.")
    except ValueError:
        print("Error: Please enter a valid number.")



mapping = {1:'db_1.csv',2:'db_2.csv',3:'db_3.csv',4:'db_4.csv',5:'db_5.csv'}
file_path = mapping[database]
while True:
    min_support_input = input("Enter minimum support as a percentage ranging 1 - 100 % (e.g., 2% -> 2): ")
    try:
        min_support = float(min_support_input) / 100
        if 0 < min_support <= 1:
            break
        else:
            print("Error: Minimum support must be between 1 and 100.")
    except ValueError:
        print("Error: Please enter a valid number.")

    # Prompt for minimum confidence with validation
while True:
    min_confidence_input = input("Enter minimum confidence as a percentage ranging 1 - 100 % (e.g., 2% -> 2): ")
    try:
        min_confidence = float(min_confidence_input) / 100
        if 0 < min_confidence <= 1:
            break
        else:
            print("Error: Minimum confidence must be between 1 and 100.")
    except ValueError:
        print("Error: Please enter a valid number.")
    


# In[2]:


def load_transactions_from_csv(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Items'].apply(lambda x: set(x.split(', '))).tolist()
    return transactions
transactions = load_transactions_from_csv(file_path)
transactions_list = [list(t) for t in transactions] 
print("Transactions from this database \n")
print(transactions)


# In[3]:


def find_frequent_itemsets_bruteforce(transactions, min_support):
    all_items = set(item for transaction in transactions for item in transaction)
    total_transactions = len(transactions)
    max_itemset_size = max([len(transaction) for transaction in transactions])
    def calc_support(itemset):
        count = sum(1 for trans in transactions if itemset.issubset(trans))
        return count / total_transactions
    
    current_itemsets = [frozenset([item]) for item in all_items]
    frequent_itemsets = []
    k = 1
    
    while current_itemsets:
        new_frequent_itemsets = [itemset for itemset in current_itemsets if calc_support(itemset) >= min_support]
        if not new_frequent_itemsets:
            break
        frequent_itemsets.extend(new_frequent_itemsets)
        if k < max_itemset_size:
            next_level_itemsets = set()
            for i in range(len(new_frequent_itemsets)):
                for j in range(i + 1, len(new_frequent_itemsets)):
                    union_set = new_frequent_itemsets[i].union(new_frequent_itemsets[j])
                    if len(union_set) == k + 1:
                        next_level_itemsets.add(union_set)
            current_itemsets = list(next_level_itemsets)
        else:
            break
        k += 1
    
    return frequent_itemsets

def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []

    def calc_support(itemset):
        return sum(1 for trans in transactions if itemset.issubset(trans)) / len(transactions)

    def calc_confidence(antecedent, consequent):
        antecedent_consequent = antecedent.union(consequent)
        return calc_support(antecedent_consequent) / calc_support(antecedent)

    for itemset in frequent_itemsets:
        for antecedent in chain.from_iterable(combinations(itemset, r) for r in range(1, len(itemset))):
            antecedent_set = frozenset(antecedent)
            consequent_set = itemset - antecedent_set
            confidence = calc_confidence(antecedent_set, consequent_set)
            if confidence >= min_confidence:
                rules.append((antecedent_set, consequent_set, calc_support(itemset), confidence))
    return rules

def run_algorithm(transactions, min_support, min_confidence, algorithm='apriori'):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    start_time = time.time()
    if algorithm == 'apriori':
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    else:
        frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"\n{algorithm.capitalize()} Execution Time: {time.time() - start_time} seconds")
    return rules

def print_rules(rules, title):
    print(f"\n{title}:")
    for rule in rules:
        antecedent, consequent, support, confidence = rule
        # Convert frozensets to sorted lists and then to strings for nicer printing
        antecedent_str = ', '.join(sorted(antecedent))
        consequent_str = ', '.join(sorted(consequent))
        print(f"Rule: {{{antecedent_str}}} => {{{consequent_str}}}, Support: {support:.2f}, Confidence: {confidence:.2f}")

def format_mlxtend_rules(rules_df):
    formatted_rules = []
    for _, row in rules_df.iterrows():
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        support = row['support']
        confidence = row['confidence']
        formatted_rules.append((antecedent, consequent, support, confidence))
    return formatted_rules





def rules_to_set(rules):
    """Convert rules into a set of immutable elements for comparison."""
    return set(
        (frozenset(antecedent), frozenset(consequent), round(support, 4), round(confidence, 4))
        for antecedent, consequent, support, confidence in rules
    )

def validate_rules_equivalence(brute_force_rules, mlxtend_rules_df):
    """
    Validates that the sets of rules from the Brute Force method and the MLxtend method
    (Apriori or FP-Growth) are exactly equivalent, without considering the order of the
    rules.

    :param brute_force_rules: Rules generated by the brute force method.
    :param mlxtend_rules_df: Rules generated by the MLxtend method (DataFrame format).
    :return: True if the sets of rules are exactly equivalent, False otherwise.
    """

    # Convert brute_force_rules to set for easy comparison
    brute_force_set = rules_to_set(brute_force_rules)

    # Convert mlxtend_rules_df DataFrame to a set of tuples
    mlxtend_set = set()
    for _, row in mlxtend_rules_df.iterrows():
        antecedent = frozenset(row['antecedents'])
        consequent = frozenset(row['consequents'])
        support = round(row['support'], 4)  # Assuming support is a column in the DataFrame
        confidence = round(row['confidence'], 4)  # Assuming confidence is a column
        mlxtend_set.add((antecedent, consequent, support, confidence))

    # Directly compare the sets
    return brute_force_set == mlxtend_set


# In[ ]:





# In[4]:


# Execute Brute Force Method
start_time = time.time()
frequent_itemsets = find_frequent_itemsets_bruteforce(transactions, min_support)
tuple_list = [tuple(fs) for fs in frequent_itemsets]
print("Frequent item sets from Brute Force Method \n")
print(tuple_list)
brute_force_rules = generate_rules(frequent_itemsets, transactions, min_confidence)
end_time = time.time()
print(f"\nBrute Force Execution Time: {end_time - start_time} seconds")
print_rules(brute_force_rules, "Brute Force Rules")


# In[5]:


# Execute Apriori
start_time = time.time()
apriori_rules_df = run_algorithm(transactions_list, min_support, min_confidence, 'apriori')
apriori_rules = format_mlxtend_rules(apriori_rules_df)
end_time = time.time()
print(f"Apriori Execution Time: {end_time - start_time} seconds")
print_rules(apriori_rules, "Apriori Rules")


# In[13]:


# Execute FP-Growth
start_time = time.time()
fp_growth_rules_df = run_algorithm(transactions_list, min_support, min_confidence, 'fpgrowth')
fp_growth_rules = format_mlxtend_rules(fp_growth_rules_df)
end_time = time.time()
print(f"FP-Growth Execution Time: {end_time - start_time} seconds")
print_rules(fp_growth_rules, "FP-Growth Rules")



# In[14]:


are_apriori_rules_equivalent = validate_rules_equivalence(brute_force_rules, apriori_rules_df)
print(f"Brute Force and Apriori rules are equivalent: {are_apriori_rules_equivalent}")

# Validate equivalence of Brute Force and FP-Growth rules
are_fp_growth_rules_equivalent = validate_rules_equivalence(brute_force_rules, fp_growth_rules_df)
print(f"Brute Force and FP-Growth rules are equivalent: {are_fp_growth_rules_equivalent}")


# In[ ]:




