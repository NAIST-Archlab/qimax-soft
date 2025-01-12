import jax.numpy as jnp
import numpy as np
import jax

@jax.jit
def custom_dot(x, y):
    return jnp.dot(x, y) ** 2

@jax.jit
def naive_custom_dot(x_batched, y_batched):
    return jnp.stack([
        custom_dot(v1, v2)
        for v1, v2 in zip(x_batched, y_batched)
    ])
    
@jax.jit
def dot_vmap(x,y):
    return jax.vmap(custom_dot, in_axes=[0, 0])(x,y)
import time


x_batched = jnp.asarray(np.random.rand(2000, 2000))
y_batched = jnp.asarray(np.random.rand(2000, 2000))
begin = time.time() 
result = naive_custom_dot(x_batched, y_batched)
print(time.time() - begin)
begin = time.time() 
result = custom_dot(x_batched, y_batched)
print(time.time() - begin)
begin = time.time() 
result = dot_vmap(x_batched, y_batched)
print(time.time() - begin)
    
    
    
    
    
    
    
    
    