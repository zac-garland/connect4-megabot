import pickle

# Load and inspect
with open('mcts7500_pool.pickle', 'rb') as f:
    data = pickle.load(f)

# Basic info
print(f"Type: {type(data)}")
print(f"Size: {len(data) if hasattr(data, '__len__') else 'N/A'}")

# Preview content
if isinstance(data, dict):
    print("\nDictionary keys:")
    for key in list(data.keys())[:10]:  # First 10 keys
        print(f"  {key}: {type(data[key])}")
        
elif isinstance(data, list):
    print(f"\nList with {len(data)} items")
    print(f"First item type: {type(data[0])}")
    print(f"First few items:")
    for i, item in enumerate(data[:3]):
        print(f"  [{i}]: {item}")
        
else:
    print(f"\nContent preview:")
    print(data)