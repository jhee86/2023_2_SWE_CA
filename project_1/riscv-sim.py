import sys


# 명령줄 인자를 리스트로 받음
arguments = sys.argv


# 첫 번째 인자는 스크립트 이름이므로 무시하고, 두 번째 인자부터 사용
if len(arguments) > 1:
    input = arguments[1]


def convert_to_little_endian(binary_data):
    # 바이너리 데이터를 리틀 엔디안 형식으로 변환
    little_endian_data = bytearray()
    for i in range(len(binary_data) - 1, -1, -1):
        little_endian_data.append(binary_data[i])
    return little_endian_data

def divide_into_32_bits(little_endian_data):
    divided_data_list = []
    current_chunk = bytearray()

    for byte in little_endian_data:
        current_chunk.append(byte)

        if len(current_chunk) == 4:  # 4바이트 (32비트)가 되면 저장
            divided_data_list.append(current_chunk.copy())
            current_chunk.clear()

    return divided_data_list


def convert_to_hex(chunk):

    hex_string = ''.join(format(byte, '02x') for byte in chunk)

    return hex_string

def sign_extend(binary_str):
    # 첫 번째 문자가 '1'인 경우 음수로 판단하고 32비트 사인 확장
    if binary_str[0] == '1':
        extended_binary = '1' * (32 - len(binary_str)) + binary_str
    else:
        extended_binary = '0' * (32 - len(binary_str)) + binary_str

    # 2진수를 10진수로 변환하여 반환
    decimal_value = int(extended_binary, 2)
    if decimal_value > 2**31 - 1:
        decimal_value -= 2**32

    return decimal_value


#이것을 디어셈블러로
def disassemble(binary_value_str): #input은 이진화된상태 str
    binary_value_str = binary_value_str.replace(" ", "")

    opcode = int(binary_value_str[-7:],2) #바로 10진수로 변환됨
    rd = int(binary_value_str[-12:-7],2)
    funct3 = int(binary_value_str[-15:-12],2)
    rs1 = int(binary_value_str[-20:-15],2)
    rs2 = int(binary_value_str[-25:-20],2)
    funct7 = int(binary_value_str[:-25],2)

    imm_i=sign_extend(binary_value_str[:12])
    imm_s=sign_extend(binary_value_str[:7]+binary_value_str[-12:-7])
    imm_b=sign_extend(binary_value_str[0]+binary_value_str[24]+binary_value_str[1:7]+binary_value_str[20:24]+"0")
    shamt=int(binary_value_str[7:12],2)
    #imm_j=sign_extend(binary_value_str[0]+binary_value_str[1:11]+binary_value_str[11]+binary_value_str[12:20]+"0")
    imm_j=sign_extend(binary_value_str[0]+binary_value_str[12:20]+binary_value_str[11]+binary_value_str[1:11]+"0")

    #imm_j = imm_j << 1
    #imm_u=sign_extend(binary_value_str[:20])
    imm_u = sign_extend(binary_value_str[0:20] + "0" * 12)

    #imm_u = imm_u << 12


#디어셈블러에서는
# 16진법 명령어를 이진화
#opcode 확인
#맨위에 저장해둔걸 가지고
#10진법으로 바꿔서 출력


    if opcode == 55: #55
        return f"lui x{rd}, {imm_u}"
    elif opcode == 23: #23
        return f"auipc x{rd}, {imm_u}"
    elif opcode == 111: #111
        return f"jal x{rd}, {imm_j}"

    elif opcode == 103: #103 #I format
        return f"jalr x{rd}, {imm_i}(x{rs1})"


    elif opcode == 3: #3 #I formt: Load
        if funct3 == 0:#0
            return f"lb x{rd}, {imm_i}(x{rs1})"
        elif funct3 == 1: #1
            return f"lh x{rd}, {imm_i}(x{rs1})"
        elif funct3 == 2:#2
            return f"lw x{rd}, {imm_i}(x{rs1})"
        elif funct3 == 4:#4
            return f"lbu x{rd}, {imm_i}(x{rs1})"
        elif funct3 == 5:#5
            return f"lhu x{rd}, {imm_i}(x{rs1})"

    elif opcode == 19: #19 #I format: 레지스터+imm 조합
        if funct3 == 0:
            return f"addi x{rd}, x{rs1}, {imm_i}"
        elif funct3 == 2:
            return f"slti x{rd}, x{rs1}, {imm_i}"
        elif funct3 == 3:
            return f"sltiu x{rd}, x{rs1}, {imm_i}"

        elif funct3 == 4:
            return f"xori x{rd}, x{rs1}, {imm_i}"
        elif funct3 == 6:
            return f"ori x{rd}, x{rs1}, {imm_i}"
        elif funct3 == 7:
            return f"andi x{rd}, x{rs1}, {imm_i}"
        elif funct3 == 1: #shift+register : func7과 shamt
            return f"slli x{rd}, x{rs1}, {shamt}"
        elif funct3 == 5: #**
            if funct7 == 0:
                return f"srli x{rd}, x{rs1}, {shamt}"

            elif funct7== 32:
                return f"srai x{rd}, x{rs1}, {shamt}"
    elif opcode == 99: #99 #SB format
        if funct3 == 0:#0
                return f"beq x{rs1}, x{rs2}, {imm_b}"
        elif funct3 == 1:#1
            return f"bne x{rs1}, x{rs2}, {imm_b}"
        elif funct3 == 4:#4
            return f"blt x{rs1}, x{rs2}, {imm_b}"
        elif funct3 == 5:#5
            return f"bge x{rs1}, x{rs2}, {imm_b}"
        elif funct3 == 6:#6
            return f"bltu x{rs1}, x{rs2}, {imm_b}"
        elif funct3 == 7:#7
            return f"bgeu x{rs1}, x{rs2}, {imm_b}"


    elif opcode == 35: #35 #S format
        if funct3 == 0:
            return f"sb x{rs2}, {imm_s}(x{rs1})"
        elif funct3 == 1:
            return f"sh x{rs2}, {imm_s}(x{rs1})"
        elif funct3 == 2:
            return f"sw x{rs2}, {imm_s}(x{rs1})"


    elif opcode == 51: #51 #R format
        if funct3 == 0:
            if funct7 == 0:
                return f"add x{rd}, x{rs1}, x{rs2}"
            elif funct7 == 32: #32
                return f"sub x{rd}, x{rs1}, x{rs2}"

        elif funct3 == 1:
                return f"sll x{rd}, x{rs1}, x{rs2}"
        elif funct3 == 2:
                return f"slt x{rd}, x{rs1}, x{rs2}"
        elif funct3 == 3:
                return f"sltu x{rd}, x{rs1}, x{rs2}"

        elif funct3 == 4:
                return f"xor x{rd}, x{rs1}, x{rs2}"
        elif funct3 == 5:
            if funct7 == 0:
                return f"srl x{rd}, x{rs1}, x{rs2}"
            elif funct7 == 32: #32
                return f"sra x{rd}, x{rs1}, x{rs2}"
        elif funct3 == 6:
            return f"or x{rd}, x{rs1}, x{rs2}"
        elif funct3 == 7:
            return f"and x{rd}, x{rs1}, x{rs2}"
    return "unknown instruction"
def riscv_sim(input): #input: 리틀 인디언이 아닌 risc-v 머신코드 bin파일
    import sys


    # 명령줄 인자를 리스트로 받음
    arguments = sys.argv

    # 첫 번째 인자는 스크립트 이름이므로 무시하고, 두 번째 인자부터 사용
    if len(arguments) > 1:
        content = arguments[1]
    with open(content, 'rb') as binary_file:
            binary_data = binary_file.read()

    little_endian_data = convert_to_little_endian(binary_data)
    divided_data_list = divide_into_32_bits(little_endian_data)

     #divided된 명령어를 가지고 반복문으로 하나씩 disassembling
    cnt=0
    for i, chunk in reversed(list(enumerate(divided_data_list))):
        binary_value_str = ' '.join(format(byte, '08b') for byte in chunk)
        decoded=disassemble(binary_value_str)
        hex_string= convert_to_hex(chunk)
        print("inst %d: %s %s"%(cnt,hex_string,decoded))
        cnt+=1


riscv_sim(input)


                    


