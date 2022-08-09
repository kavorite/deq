import jax
import jax.numpy as jnp


def pos_emb(period, width, depth):
    inv = jnp.power(period, jnp.arange(width) / width)
    emb = jnp.empty([width])
    emb = emb.at[::2].set(jnp.sin(depth / inv[::2]))
    emb = emb.at[1::2].set(jnp.cos(depth / inv[1::2]))
    return emb


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    d = 128
    A = jax.vmap(pos_emb, in_axes=(None, None, 0), out_axes=1)(d, d, jnp.arange(d))
    plt.imshow(A)
    plt.show()
