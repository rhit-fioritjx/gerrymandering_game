# Mathematical Observations Resulting from the Creation of the Gerrymandering Game
## Introduction
Despite being a simple game there are acually some interesting mathematical implications that have resulted from its implementation. The main implication of this work is the minimum population required to win a plurality of districts under single representative plurality elections. This condition was found during the implementation of the initial solution generation step of the puzzle initialization. In this short piece of documentation I will explain the origin of this condition, its real world implications, and how this can guide some districting policy.
## The Minimum
To implement the puzzles of the gerrymandering game, I needed to calculate the minimum number of voters required to win a plurality of districts under a plurality election. 
This was to ensure the driving condition of the game, that a minority party win a plurality of districts. 
In this calculation there are three key variables. 
The voting population of the area being districted ($n\geq k,p$), the number of districts ($k\geq1$), and the number of parties being voted for ($p\geq2$). 
This minimum number of voters to overall win a plurality of districts under plurality conditions is the product of the number of districts which must be won and the number of voters in each of those districts that must vote for the winning party. 

Both of these follow the same formulation, as both are winning plurality elections, one just has $k$ total "voters" and the other has $n/k$ "voters". 
For notational simplicity I will just run through the formulation for the number of districts which need to be won. 

In order to win a plurality among $k$ districts, a party must have won the most districts out of all $p$ districts. 
Let's call the number of districts which the winning party wins $i$. 
Becaulse the remaining $k-i$ districts may not be divisible by the remaining $p-1$ parties we take the ceiling to get the greatest number of districts won amongst our reasonably even division. 
$$i>\lceil\frac{k-i}{p-1}\rceil$$
Because $i$ is an integer $i-1\geq\lceil\frac{k-1}{p-1}\rceil$ and since $\lceil x\rceil\geq x$ we can remove the ceiling from our fraction yeilding:
$$i-1\geq\frac{k-i}{p-1}$$ 
We can now rearange the terms to find an inequality with $i$ on only one side, by multipling by the positive $p-1$ and then moving around a few terms.
$$i\geq\frac{k+p-1}{p}$$
Because we want $i_{min}$ which is a minimal integer solution for $i$ we can solve from this inequality.
$$i_{min}=\lceil\frac{k+p-1}{p}\rceil$$
Validating this against our understanding of plurality voting we see that as the number of representatives increases the minimum victories required increases. 
We also see that as the number of parties increases the minimum required decreases. 
Both of these are consistent with our understanding of plurality elections which provides additional validation to this formulation.

Taking the product of the number of voters required to win a district and the number of districts to win the overall plurality we get the following formulation for $w_{min}$ the minimum number of voters required to win the overall plurality.
$$w_{min}=\lceil\frac{k+p-1}{p}\rceil\lceil\frac{n/k+p-1}{p}\rceil$$
To explore how each of these variables impacts this minimum threshhold more explicitly than with a qualitative exploration of the graph we would like to calculate $\nabla w_{min}$. 
Because of the ceiling functions however this gradient is not calculable. 
While there may be better ways to approach this I do not have a strong enough mathematical background to know the correct approaches, and so I will simply be dropping the ceilings.
This loses some substantial detail but is able to retain some overall trends.
We will call this approximation of $w_{min}$ $\^{w}$.
$$\nabla\^{w}=(\frac{\partial\^w}{\partial n},\frac{\partial\^w}{\partial k},\frac{\partial\^w}{\partial p})=\left(\frac{k+p-1}{kp^2},\frac{(p-1)(k^2-n)}{p^2k^2},-\frac{(n+k^2-2k)p+(2k-2)n-2k^2+2k}{kp^3}\right)$$
While this does not capture all the detail of the original function, it can tell us a few things about the relationship of these variables. 

Starting with n, we can see that because $k>0$ and $p>1$ the partial derivative with respect to n is always positive, so increasing the voting population will always increase the difficulty of gerrymandering. 

Next for k we can do a similar analysis, $p-1>0$ and $k^2p^2>0$ so $\frac{\partial\^w}{\partial k}$ has the same sign as $k^2-n$. This tells us that generally when $k<\sqrt{n}$ it will be harder to gerrymander with fewer districts, and when it is greater, it will be more difficult to gerrymander with more. Looking at the graphs, this trend is not explicitly true, as there may be some jumps in the opposite directions of the trend, however it does give an impression of the general shape.

For p the analysis is a bit more complicated but is still doable. $kp^3>0$ so the sign is dependent on the sum in the numerator. For this gradient to be positive $2k^2>(n+k^2-2k)p+(2k-2)n+2k$. Because the minimum of $k^2-2k$ on the integers is $-1$ and $n\geq p \geq 2$, $(n+k^2-2k)>0$. So the right expression will be minimized under minimal p. Thus $p=2$ and $2k^2 > 2n + 2k^2 - 4k + 2kn - 2n + 2k$. Simplifying we get $0>2nk-2k$ which is only true when n is less than k. Therefor $\^w$ is always decreasing with respect to p. This means that the more competitive parties we have the easier it is to gerrymander a districted plurality. Additionally because the internal value in both terms is explicitly decreasing, we do not get any of the jumps which are present with respect to k.

## Implications
While the real world implications were breifly discussed in the analysis section of this document I want to make these implications explicit. 

The total population was found to be directly correlated to the minimum voters required to gerrymander a district. 
This means that the greater the population of an area subject to gerrymandering the more difficult it is to do so. 
This much like previous results about plurality voting exemplify the importance of voter turnout to functional plurality based democracies. 

The number of parties on the other hand was found to be inversely proportional to the minimum. 
This parellels the spoiler effect seen in other analysis of the plurality voting system, where increasing the number of policies decreases expected efficiency of the system.
The negative impacts of increasing the number of policies are in competition with the increase to the breadth of political expression avaliable, leading to an interesting optimization problem. 
With the multiple layers of plurality voting present in a gerrymandered system we can actually expect the spoiler effect to be multiplied back on itself increasing its overall effect.
We can actually see this by taking $\frac{\partial\^w}{\partial pk}=\frac{(n-k^2)(p-2)}{k^2p^3}$ which shows that while $k<\sqrt{n}$ increasing $k$ makes $\^w$ decrease more quickly with respect to $p$ when $p$ is greater than 2. 
For almost all practical applications $k$ will be less than $\sqrt{n}$ so minimizing the number of districts minimizes the redoubling of the spoiler effect.

Finally k was found to have a negative relation with the size when k was small $k<\sqrt{n}$ and a positive when k is large. 
In the real world districting is only really applied when there are large numbers of voters, and the number of districts chosen is generally less than $\sqrt{n}$. 
Because of this smaller numbers of districts generally results in less gerrymanderable political systems. 
Interestingly this can mean that having few districts is actually a good thing for minimizing the the amount that a region can be gerrymandered. 
This parellels the conflict present in the number of parties, where a greater number of districts alows for more diversity in the representatives, but also (generally) decreases the minimum number of voters needed to win a representative plurality. 
## Guiding Districting Policy
With the objective of minimizing the amount a region can be gerrymandered, the math presented here would suggest decreasing the number of parties and districts and increasing the voting population. 
Decreasing the number of parties is not reccomended by the author as it limits the spectrum of political expression. 
Decreasing the number of districts is also somewhat problematic as it decreases the diversity of perspectives which may be present in the next layer of government. 
The cube root law may provide a reasonable comprimise between this expression and the risk of gerrymandering.
Increasing the size of the voting population on the other hand is somewhat encouraged by the author as encouraging people to vote and allowing marginalized groups to be represented has relatively little drawback.

One comprimise that we see used in the real world with respect to districting is to have certain decisions be put to a broader plurality vote such as how most states determine their electoral college votes based on the states popular vote rather than by districts. 
Another variant of this idea is allowing the legislature to make short term decisions which can then be approved of or rejected at a later time by the constituency. 
These systems are useful as they bypass the issues with double pluralities while still allowing for the legislature to function as a governmental system.
## Future Work
In the future I hope to extend this game to allow for districts with size tolerances, non deterministic voters, and other such mechanics. As these features are introduced they will add interesting complications to the mathematics presented here, which may have some more interesting implications.