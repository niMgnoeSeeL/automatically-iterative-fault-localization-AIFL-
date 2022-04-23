
I(s[n+1]) 이려면
    - I(s[n]) 일 경우, e[n]이 s[n]을 mask 하지 않아야 하고 (not M(e[n]))
    - I(s[n]) 가 아닐 경우, e[n]이 faulty 하며 (F(e[n])) e[n]이 s[n]을 pollute 시켜야 한다 (P(e[n], s[n])).

I(s)
M(e)
F(e)
P(e, s)

만약 CC가 발생했다면,
1) pollution이 안 되었거나, 
2) masking이 되었을 것.

CC가 발생했는지 모르고, 가장 buggy하다고 했던 e가 있었을 때, 만약 F(e) = 0 이라고 얘기 한다면, 나며지 F(e)가 알맞게 update 되는가?
- 실제로는 다른 e에서 infected 되었다고 유추해야 하고,
- 그 다른 e를 실행하는 passing execution에선, pollution이 일어나지 않았거나, masking이 일어났다고 판단해야 한다.
  - **pollution이 일어나거나 masking이 될 확률이 update 되어야 하는게 맞나? 아니면 constant이어야 하나?** -- 이거 생각해보고
- **siemens suite에 대해서 update가 어떻게 작동할 지 생각해보고**


- simulated annealing으로 posteria estimate 해 놨는데, 이게 size 커졌을 때도 잘 작동할까?