# ppo-lunar-landar
A working PPO implementation for Lunar Lander v3

Does not use entropy (i forgot to implement it)

Explanation:
When running the simulation, the agent gathers information and has <last_epoch> and <this_epoch> information stored. Every epoch, the value network will train off temporal difference error (of last epoch), while the policy network's loss is a clipped loss (PPO loss), utilising r_value.

![image](https://github.com/user-attachments/assets/a39338c3-05ca-4766-935e-fc5aaa23e7db)
