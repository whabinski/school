@ ---------------------------------------
@   Data Section
@ ---------------------------------------
.data
new_val: .word 0xBD5B7DDE
@ ---------------------------------------
@   Code Section
@ ---------------------------------------
.text
.global main
.extern int_out

main:
  @ push the return address (lr) and a 
  @ dummy register (ip) to the stack 
  push {ip, lr}    

  @ load the address of variable new_val into r1
  ldr r1, =new_val
  @load the value at memory address r1 into r0  
  ldr r0, [r1]
  @ r0 = r0 right shifted by 1 bit
  mov r0, r0, ASR #1

  @ branch to int_out, passing r0 as argument
  bl int_out
  
  @ pop the return address into the program counter
  pop {ip, pc}