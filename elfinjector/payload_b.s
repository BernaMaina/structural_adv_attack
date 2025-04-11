section .text
  global _start
  global _loc0
  global _loc1
  global _loc2
  
_start:  
  cmp rax, rax
  jnz _loc0
  
_loc0:
  cmp rbx, rbx
  jnz _loc1
  
_loc1:
  cmp rcx, rcx
  jnz _loc2
  
_loc2:
  cmp rdx, rdx
  jnz _start
  jmp -0x0000
