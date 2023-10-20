# boolean binary string adder

def rjust_length(s1, s2, fill='0'):
    l1, l2 = len(s1), len(s2)
    if l1 > l2:
        s2 = s2.rjust(l1, fill)
    elif l2 > l1:
        s1 = s1.rjust(l2, fill)
    return (s1, s2)

# def get_input():
#     bits_a = input('input your first binary string  ')
#     bits_b = input('input your second binary string ')
#     return rjust_length(bits_a, bits_b)

def xor(bit_a, bit_b):
    A1 = bit_a and (not bit_b)
    A2 = (not bit_a) and bit_b
    return int(A1 or A2)

def half_adder(bit_a, bit_b):
    return (xor(bit_a, bit_b), bit_a and bit_b)

def full_adder(bit_a, bit_b, carry=0):
    sum1, carry1 = half_adder(bit_a, bit_b)
    sum2, carry2 = half_adder(sum1, carry)
    return (sum2, carry1 or carry2)

def binary_string_adder(bits_a, bits_b):
    if bits_a[0:2] == '0b':  # python binary string
        bits_a = bits_a[2::]
    if bits_b[0:2] == '0b':  # python binary string
        bits_b = bits_b[2::]
    bits_a, bits_b = rjust_length(bits_a, bits_b)
    carry = 0
    result = ''
    for i in range(len(bits_a)-1 , -1, -1):
        summ, carry = full_adder(int(bits_a[i]), int(bits_b[i]), carry)
        result += str(summ)
    result += str(carry)
    return result[::-1]

# def main():
#     bits_a, bits_b = get_input()
#     print('1st string of bits is : {}, ({})'.format(bits_a, int(bits_a, 2)))
#     print('2nd string of bits is : {}, ({})'.format(bits_b, int(bits_b, 2)))
#     result = binary_string_adder(bits_a, bits_b)
#     print('summarized is         : {}, ({})'.format(result, int(result, 2)))

# if __name__ == '__main__':
#     main()