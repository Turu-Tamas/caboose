 This is project for me to try CUDA programming. The problem was to find caboose numbers.
 A positive integer x is called a caboose number if and only if for all integers i such that 1 < i < x, the number i^2 + x is prime
 (see [this video](https://www.youtube.com/watch?v=gM5uNcgn2NQ) from Numberphile).

 This code checked 10^12 numbers in ~34mins on a 4070.

 I ran it on linux with:
 ```nvcc -arch native caboose.cu --run```
