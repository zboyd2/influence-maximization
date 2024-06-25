# Influence Maximization
This project uses several algorithms of varying complexity to simulate the spread of influence in a community. Some of the more confusing algorithms will be explained in greater depth below.

## `get_influence(graph_laplacian, config)`
### Explanation
This function is used to calculate the total influence each player has in the game. It is used within the bot algorithms. It takes `graph_laplacian` (simply the graph laplacian of the game graph) and `config` (an array of the nodes selected by the players, starting with team 0's first choice and alternating teams until the end) as parameters. 

For ease of explanation, suppose that $n$ represents the number of nodes in the game graph and $c$ represents the length of the `config` array (or the number of turns that have taken place so far). Also note that the code creates an array named `compliment` that contains all of the nodes not owned by either player.

This function uses `np.linalg.lstsq` to solve the following matrix equation: $$L\vec{x} = -B\vec{t}$$ $L$, $\vec{x}$, $B$, and $\vec{t}$ are defined as follows:

$L$ is the $(n-c) \times (n-c)$ laplacian of the subgraph. The subgraph is created by taking the orignal graph laplacian and removing all of the rows and columns associated with nodes controlled by either player. In other words, it removes the columns and rows associated with the nodes in `config`. 

$\vec{x}$ is a vector containing variables representing the influence of the nodes that are _not_ controlled by either team. These are the variables we are solving for in the matrix equation. If `compliment` were to be $[5, 7, 9]$, then $\vec{x}$ would be $[x_{5}, x_{7}, x_{9}]$. The whole purpose of this function is determining how much influence each of these uncontrolled nodes has.

$B$ is the $(n-c) \times c$ boundary block matrix. Its rows represent the nodes that are not controlled by either team while its columns represent the nodes found in `config`. This can make the indexing a little confusing. The matrix entry $B_{ij}$ is $-1$ if there is a connection between the $i$-th node in `compliment` and the $j$-th node in `config`. Otherwise, it is $0$. See the example below for a clearer demonstration of this.

$\vec{t}$ is a vector of length $c$ containing a sequence of alternating 0s and 1s. If you are calculating influence from `team_0`'s perspective, the first entry of this vector is $1$ (because `team_0` owns every other node in `config` starting from the first entry). If you are calcuating influence from `team_1`'s perspective, the first etnry of this vector is $0$.

The result of solving this matrix equation is a vector of influence values (between $0$ and $1$) for each node in `compliment`. To calculate a team's overall influence, you simply sum the values of this vector and add it to the number of nodes that team controls (recall that nodes controlled by a team automatically have an influence of $1$).

It is important to note that the sum of the two team's overall influence is **only guaranteed to equal $n$ when the game graph is connected**. This is why we are unable to take the easy route and subtract one team's influence from $n$ to calculate the other team's influence.

### Example
Even after breaking down the matrix equation that drives this algorithm, it may still feel confusing. Stepping through the following example should help build your intuition of how this algorithm works.

Suppose we are passed in $[1, 2, 3]$ as our `config` array and the following `graph_laplacian`

$$\begin{bmatrix}
3 & -1 & 0 & -1 & -1 \\
-1 & 2 & 0 & 0 & -1 \\
0 & 0 & 2 & -1 & -1 \\
-1 & 0 & -1 & 3 & -1 \\
-1 & -1 & -1 & -1 & 4
\end{bmatrix}$$

In this example, $n=5$, $c=3$, and `compliment` is $[0, 4]$. Keep in mind that we will be using **zero indexing for everything**, so the nodes are labeled from $0$ to $4$.

The function starts by computing the values it will need to solve the matrix equation $L\vec{x}=-B\vec{t}$.

The laplacian of the subgraph $L$ is computed by removing the rows and columns of the `graph_laplacian` associated with the nodes in `config`. In this case, we remove rows $1$, $2$, and $3$, giving us the following:

$$L = \begin{bmatrix}
3 & -1 \\
-1 & 4
\end{bmatrix}$$

The boundary block matrix $B$ is computed with the following fancy python array indexing expression: `graph_laplacian[compliment, :][:, config]`.

$$B = \begin{bmatrix}
-1 & 0 & -1 \\
-1 & -1 & -1
\end{bmatrix}$$

Making sense of this matrix is a little challenging. You can think of the rows representing the unclaimed nodes found in the `compliment` array and the columns representing the nodes in the `config` array. The reason the entry $B_{1,0}$ is $0$ is because there is no connection between the first node in `compliment` (node $0$) and the second node in `config` (node $2$). The rest of the entries are $-1$ because there connections between their corresponding nodes.

The defintion of $\vec{t}$ varies depending on which team we are calculating influence for. Let's start by calculating `team_0`'s influence, represented in the code by the variable `influence_0`. Because `team_0` gets the first move of the game, they own every other node in the `config` array starting from the first entry. We represent this by defining $\vec{t}$ as $[1, 0, 1]$.

The vector $\vec{x}$ is not explicitly definined the in code, but it is helpful to think of in the context of solving the matrix equation. For this example, $\vec{x} = [x_0, x_4]$. These are the two nodes that we are trying to calculate the influence of. We already know the influence of nodes $1$, $2$, $3$, since they are already controlled by a team and necessarily have an influence of $1$ (or $0$ depending on team perspective). By solving for these two unknowns, we can determine how much all five of the nodes are contributing to the influence of each team.

Our matrix equation $L\vec{x} = -B\vec{t}$ now looks like this:

$$
\begin{bmatrix}
3 & -1 \\
-1 & 4
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_4
\end{bmatrix}
=
-\begin{bmatrix}
-1 & 0 & -1 \\
-1 & -1 & -1
\end{bmatrix}
\begin{bmatrix}
1 \\
0 \\
1
\end{bmatrix}
$$

Simplifying gives us the following augmented matrix:

$$
\left[ 
    \begin{array}{cc|c}
        3 & -1 & 2 \\
        -1 & 4 & 2 \\
    \end{array}
\right]
$$

Solving this system of linear equations reveals that $x_0 = 0.\overline{90}$ and $x_4 = 0.\overline{72}$. Because we are calculating `influence_0`, these quantities tell us that node 0 contributes roughly $0.9$ to `team_0`'s overall influence and node 4 contributes roughly $0.7$ to `team_0`'s influence. To calculate `team_0`'s overall influence, we add up the influence of each of the nodes.

$$
\texttt{influence\_0} = x_0 + x_1 + x_2 + x_3 + x_4  \\
\texttt{influence\_0} = 0.\overline{90} + 1 + 0 + 1 + 0.\overline{72}
$$

$$
\texttt{influence\_0} = 3.\overline{63}
$$

Recall that from `team_0`'s perspective, nodes that are controlled by `team_0` have an influence of $1$ and nodes that are controlled by `team_1` have an influence of $0$. Because this particular network is connected, `influence_1` can be calculated simply by subtracting `influence_0` from $n$. Alternatively (and necessarily in cases where the game network is *not* connected) `influence_1` can be calculated by solving the same matrix equation above, this time with $t = [0, 1, 0]$ because `team_1` controls every other node in `config` starting with second entry. This results in `influence_1` $= 1.\overline{36}$.