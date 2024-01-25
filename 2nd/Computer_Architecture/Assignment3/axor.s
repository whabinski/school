@ ---------------------------------------
@   Data Section
@ ---------------------------------------
.data
@ ---------------------------------------
@   Code Section
@ ---------------------------------------
.text
.global axor

axor:
  @ push the return address (lr) and a 
  @ dummy register (ip) to the stack 
  push {ip, lr}    

  EOR r0, r0, r1
  
  pop {ip, pc}
