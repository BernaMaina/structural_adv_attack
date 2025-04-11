
 
 section .text
  global _start
  global _loc
  
_start:  
  cmp rax, rax
  jnz _loc
  
_loc:
  jmp -0x0000