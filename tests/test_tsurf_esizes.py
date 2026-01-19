from xali_tools.io.tsurf_filter import load_tsurf, save_tsurf
import numpy as np
import tempfile
import os

# Load the test file
print('=== Loading two-triangles.gcd ===')
data = load_tsurf('tests/two-triangles.gcd')

print('Surface name:', data.name)
print('Properties:', list(data.properties.keys()))
print()

# Check U (should be 3D vector)
U = data.properties['U']
print('U shape:', U.shape)
print('U values:')
print(U.flatten())
print()

# Check scalar properties
for name in ['a', 'b']:
    val = data.properties[name]
    print(f'{name} shape: {val.shape}, values: {val}')

print()
print('=== Saving to temp file ===')
with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
    filepath = f.name

save_tsurf(data, filepath)

with open(filepath, 'r') as f:
    content = f.read()
print('Saved content:')
print(content)

print('=== Reloading saved file ===')
reloaded = load_tsurf(filepath)
print('Properties:', list(reloaded.properties.keys()))
print('U shape:', reloaded.properties['U'].shape)
print('U matches:', np.allclose(reloaded.properties['U'], data.properties['U']))
print('a matches:', np.allclose(reloaded.properties['a'], data.properties['a']))
print('b matches:', np.allclose(reloaded.properties['b'], data.properties['b']))

os.unlink(filepath)
print()
print('=== All tests passed! ===')