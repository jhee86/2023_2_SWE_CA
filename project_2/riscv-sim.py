import sys

MEMORY_START = 0x10000000
MEMORY_END = 0x1000FFFF

registers = ["0"*32] * 32 #레지스터 개수 32개,32bit 이진문자열 0으로 초기화
memory = [0] * 65536  # 64KB memory
pc = 0 #pc도 이진 바이너리로 해야할까..? 왜냐면 분기 점프할 때 pc.. 이건 다시 생각해보자.
registers[0] ="0"*32 #x0 is fixed to zero.


# 시작 주소와 끝 주소 정의
start_address = 0x10000000
end_address = 0x1000FFFF

# 데이터를 저장할 빈 사전 생성
data_memory = {}
initial_value ="11111111"
for address in range(start_address, end_address + 1):
    data_memory[address] = initial_value

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
#바이너리 문자열 반환
    return extended_binary,decimal_value



#이것을 디어셈블러로
def disassemble_and_execute(binary_value_str): #input은 이진화된상태 str 레지스터에 저장된 상태도 binary str 상태여야 함

    global MEMORY_START, MEMORY_END, registers, memory, pc, dc,data_memory
    registers[0] ="0"*32 #x0 is fixed to zero.

    binary_value_str = ''.join(binary_value_str)
    binary_value_str = binary_value_str.replace(" ", "")

    opcode = int(binary_value_str[-7:],2) #바로 10진수로 변환됨
    rd = int(binary_value_str[-12:-7],2)
    funct3 = int(binary_value_str[-15:-12],2)
    rs1 = int(binary_value_str[-20:-15],2)
    rs2 = int(binary_value_str[-25:-20],2)
    funct7 = int(binary_value_str[:-25],2)

    rs1_value = int(registers[rs1], 2)
    rs2_value = int(registers[rs2], 2)

    #unsigned
#sign

    #int 상태
    non_used,imm_i=sign_extend(binary_value_str[:12])
    shamt=int(binary_value_str[7:12],2)

    #binary str 상태
    non_used,imm_s=sign_extend(binary_value_str[:7]+binary_value_str[-12:-7])
    non_used,imm_b=sign_extend(binary_value_str[0]+binary_value_str[24]+binary_value_str[1:7]+binary_value_str[20:24]+"0")
    non_used,imm_j=sign_extend(binary_value_str[0]+binary_value_str[12:20]+binary_value_str[11]+binary_value_str[1:11]+"0")
    imm_u,non_used=sign_extend(binary_value_str[:20])
    #non_used,rs1_value= sign_extend(registers[rs1])
    #non_used,rs2_value=sign_extend(registers[rs2])
    if opcode == 55: #55
        registers[0] ="0"*32
        binary_string=imm_u #2진 문자
        binary_value = int(binary_string, 2)
        shifted_value = binary_value << 12
        result = shifted_value & 0xFFFFFFFF

        registers[rd]= format(result, '032b')
        #print("registers[rd]",result)
        #print("현재 pc value:",pc)
        pc += 4

        return f"lui x{rd}, {imm_u}"


    elif opcode == 23: #23
        registers[0] ="0"*32
        pc_value = pc # PC 레지스터 (예를 들면, x32)
        binary_string=imm_u #2진 문자
        imm_u = int(binary_string, 2)

        result = pc_value + (imm_u << 12)
        result = result & 0xFFFFFFFF
        registers[rd] = format(result, '032b')
        #print("현재 pc value:",pc)
        pc += 4

        return f"auipc x{rd}, {imm_u}"

    elif opcode == 111: #111
        registers[0] ="0"*32

        result=pc+4
        registers[rd] = format(result, '032b')
        #print("현재 pc value:",pc)
        pc+=imm_j


        return f"jal x{rd}, {imm_j}"

    elif opcode == 103: #103 #I format
        registers[0] ="0"*32
        result=pc+4
        registers[rd] = format(result, '032b')
        #target_address = (rs1_value + imm_i) & 0xFFFFFFFE  # 홀수 비트를 제외한 주소

        target_address=rs1_value + imm_i
        #print("현재 pc value:",pc)
        pc = target_address


        return f"jalr x{rd}, {imm_i}(x{rs1})"


    elif opcode == 3: #3 #I formt: Load
        registers[0] ="0"*32
        if funct3 == 2:#2
            addr=rs1_value+imm_i
            result=0
            if rs1_value == 0x20000000:
                #user_input = int(input("\nEnter a number: "))
                user_input = int(input())
                registers[rd] = format(user_input, '032b')
            else:
                for i in range(4):
                    data_byte = data_memory.get(addr + i, 0)  # 주어진 주소에서 1바이트 데이터 읽기, 주소가 없을 경우 0으로 초기화
                    data_byte=int(data_byte, 2)
                    #print(data_byte)
                    #print(type(data_byte))
                    result = (result << 8) | data_byte  # 데이터 변수에 1바이트씩 추가하여 연결
                #print("data:",result)
                registers[rd] = format(result, '032b')
            #print("현재 pc value:",pc)
            pc += 4

            return f"lw x{rd}, {imm_i}(x{rs1})"


    elif opcode == 19: #19 #I format: 레지스터+imm 조합
        registers[0] ="0"*32
        if funct3 == 0:
            result= rs1_value+imm_i
            result = result & 0xFFFFFFFF
            registers[rd] = format(result, '032b')
            #print("현재 pc value:",pc)
            pc += 4

            return f"addi x{rd}, x{rs1}, {imm_i}"

        elif funct3 == 2:
            if rs1_value < imm_i:
                registers[rd] = "00000000000000000000000000000001"
            else:
                registers[rd] = "0"*32
            #print("현재 pc value:",pc)
            pc += 4

            return f"slti x{rd}, x{rs1}, {imm_i}"

   
        elif funct3 == 4:
            result = rs1_value ^ imm_i
            result = result & 0xFFFFFFFF
            registers[rd] = format(result, '032b')  # 32비트 이진수로 변환하여 저장
            #print("현재 pc value:",pc)
            pc += 4


            return f"xori x{rd}, x{rs1}, {imm_i}"


        elif funct3 == 6:
            result = rs1_value | imm_i
            result = result & 0xFFFFFFFF
            registers[rd] = format(result, '032b')
            #print("현재 pc value:",pc)
            pc += 4

            return f"ori x{rd}, x{rs1}, {imm_i}"

        elif funct3 == 7:
            result = rs1_value & imm_i
            result = result & 0xFFFFFFFF
            registers[rd] = format(result, '032b')
            #print("현재 pc value:",pc)
            pc += 4

            return f"andi x{rd}, x{rs1}, {imm_i}"

        elif funct3 == 1: #shift+register : func7과 shamt
            result = rs1_value << shamt
            result = result & 0xFFFFFFFF  # 32비트로 제한
            registers[rd] = format(result, '032b')
            #print("현재 pc value:",pc)
            pc += 4

            return f"slli x{rd}, x{rs1}, {shamt}"

        elif funct3 == 5: #**
            if funct7 == 0:
                result = rs1_value >> shamt
                result = result & 0xFFFFFFFF
                result = format(result, '032b')
                registers[rd]=result
                #print("현재 pc value:",pc)
                pc += 4

                return f"srli x{rd}, x{rs1}, {shamt}"


            elif funct7== 32:
                original_value = rs1_value
                sign_bit = original_value & 0x80000000  # 가장 상위 비트(부호 비트) 추출

                shifted_value = original_value >> shamt


                # 부호 확장을 수행
                if sign_bit:  # 원래 값이 음수일 경우
                    sign_extension = (0xFFFFFFFF << (32 - shamt))
                    result= shifted_value | sign_extension
                    result = result & 0xFFFFFFFF
                    result = format(result, '032b')
                    registers[rd]=result

                else:
                    result = shifted_value
                    result = result & 0xFFFFFFFF
                    result = format(result, '032b')
                    registers[rd]=result
                #print("현재 pc value:",pc)
                pc += 4


                return f"srai x{rd}, x{rs1}, {shamt}"


    elif opcode == 99: #99 #SB format
            registers[0] ="0"*32
            non_used,rs1_value= sign_extend(registers[rs1])
            non_used,rs2_value=sign_extend(registers[rs2])
            if funct3 == 0:#0
                if rs1_value == rs2_value:
                    #print("현재 pc value:",pc)
                    pc += imm_b
                    # 분기 (PC에 오프셋을 더함)
                else:
                    pc+=4

                return f"beq x{rs1}, x{rs2}, {imm_b}"

            elif funct3 == 1:#1
                if rs1_value != rs2_value:
                    #print("현재 pc value:",pc)
                    #print("분기시작, pc:",pc)
                    pc += imm_b  # 분기 (PC에 오프셋을 더함)
                else:
                    pc+=4
                return f"bne x{rs1}, x{rs2}, {imm_b}"


            elif funct3 == 4:#4
                if rs1_value < rs2_value:
                    #print("현재 pc value:",pc)
                    pc += imm_b  # 분기 (PC에 오프셋을 더함)
                else:
                    pc+=4
                return f"blt x{rs1}, x{rs2}, {imm_b}"
            
            elif funct3 == 5:#5
                #print("rs1:",rs1_value)
                #print("rs2:",rs2_value)
                if rs1_value >= rs2_value:
                    #print("현재 pc value:",pc)
                    pc += imm_b
                else:
                    pc+=4
                return f"bge x{rs1}, x{rs2}, {imm_b}"



    elif opcode == 35: #35 #S format
        registers[0] ="0"*32
        if funct3 == 2:
            addr = imm_s+rs1_value
            binary_value_str = registers[rs2]
            data_memory_counter=0
            for i in range(0, len(binary_value_str), 8):
                    byte_str = binary_value_str[i:i+8]  # 8자씩 잘라서 이진 문자열 생성
                    #print("byte_str:",byte_str)
                    data_memory[addr + data_memory_counter] = byte_str
                    data_memory_counter+=1
                    if rs1_value == 0x20000000:
                        print(chr(int(byte_str, 2)), end='')
            #print("현재 pc value:",pc)
            pc += 4

            return f"sw x{rs2}, {imm_s}(x{rs1})"

    elif opcode == 51: #51 #R format
            registers[0] ="0"*32    
            if funct3 == 0:
                if funct7 == 0:

                    result = rs1_value + rs2_value
                    result = result & 0xFFFFFFFF
                    registers[rd] = format(result, '032b')
                    #print("현재 pc value:",pc)
                    pc += 4


                    return f"add x{rd}, x{rs1}, x{rs2}"
                elif funct7 == 32: #32

                    result = rs1_value - rs2_value
                    result = result & 0xFFFFFFFF
                    registers[rd] = format(result, '032b')
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"sub x{rd}, x{rs1}, x{rs2}"


            elif funct3 == 1:
                    shift_amount = rs2_value & 0x1F
                    # 쉬프트 연산 수행
                    result = rs1_value << shift_amount
                    result = result & 0xFFFFFFFF
                    registers[rd] = format(result, '032b')
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"sll x{rd}, x{rs1}, x{rs2}"

            elif funct3 == 2:
                    if registers[rs1] < registers[rs2]:
                        registers[rd] = "00000000000000000000000000000001"
                    else:
                        registers[rd] = "0"*32
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"slt x{rd}, x{rs1}, x{rs2}"

            elif funct3 == 4:

                    rs1_value = int(registers[rs1], 2)
                    rs2_value = int(registers[rs2], 2)

                    result=rs1_value^rs2_value
                    result = result & 0xFFFFFFFF
                    registers[rd] = format(result, '032b')
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"xor x{rd}, x{rs1}, x{rs2}"
            elif funct3 == 5:
                if funct7 == 0:
                    shift_amount = rs2_value & 0x1F

                    # 쉬프트 연산 수행
                    result = rs1_value >> shift_amount

                    result = result & 0xFFFFFFFF
                    registers[rd] = format(result, '032b')
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"srl x{rd}, x{rs1}, x{rs2}"


                elif funct7 == 32: #32
                    original_value = rs1_value
                    sign_bit = original_value & 0x80000000  # 가장 상위 비트(부호 비트) 추출
                    shamt = rs2_value & 0x1F

                    shifted_value = original_value >> shamt

                    # 부호 확장을 수행
                    if sign_bit:  # 원래 값이 음수일 경우
                        sign_extension = (0xFFFFFFFF << (32 - shamt))
                        result= shifted_value | sign_extension
                        result = result & 0xFFFFFFFF
                        result = format(result, '032b')
                        registers[rd]=result

                    else:
                        result = shifted_value
                        result = result & 0xFFFFFFFF
                        result = format(result, '032b')
                        registers[rd]=result
                    #print("현재 pc value:",pc)
                    pc += 4

                    return f"sra x{rd}, x{rs1}, x{rs2}"


            elif funct3 == 6:
                rs1_value = int(registers[rs1], 2)
                rs2_value = int(registers[rs2], 2)
                result = rs1_value | rs2_value
                result = result & 0xFFFFFFFF
                registers[rd] = format(result, '032b')
                #print("현재 pc value:",pc)
                pc += 4

                return f"or x{rd}, x{rs1}, x{rs2}"

            elif funct3 == 7:

                rs1_value = int(registers[rs1], 2)
                rs2_value = int(registers[rs2], 2)
                result = rs1_value & rs2_value
                result = result & 0xFFFFFFFF

                registers[rd] = format(result, '032b')
                #print("현재 pc value:",pc)
                pc += 4

                return f"and x{rd}, x{rs1}, x{rs2}"
    return "unknown instruction"



def print_registers(registers):
    for i, value in enumerate(registers):
        value = int(value, 2)
        print(f'x{i}: 0x{value:08x}')
    #print("registers:",registers)


def simulate(instr_count=None, intr_count=None):
    global MEMORY_START, MEMORY_END, registers, memory, pc,dc
    registers[0] ="0"*32
    while True:
            if instr_count is not None and instr_count == 0:
                break

            if intr_count is not None and intr_count == 0:
                break



            binary_list = memory[pc:pc + 4]  # memory에서 4바이트 데이터를 가져옴
            #print("binary_list: (int 0이 나오면 멈추게 설정할 것임):",binary_list)
            if binary_list == [0,0,0,0]:

                break

            binary_value_str = ''.join(binary_list)

            # 결과 출력
            #print("메모리에서 가져옴:",binary_value_str)
             #이걸 다시 스트링으로 바꾸기
            hex_string = format(int(binary_value_str, 2), 'X')
            registers[0] ="0"*32
            decoded=disassemble_and_execute(binary_value_str)
            registers[0] ="0"*32
            
            ###########명령어 다 보여 주는 print#######################
            #print("%s %s"%(hex_string,decoded))
            if len(binary_value_str) < 4:
                break

            if instr_count is not None:
                instr_count -= 1

            if intr_count is not None:
                intr_count -= 1
    print_registers(registers) #레지스터 최종상태 출력



def riscv_sim_simulater():  #input: 리틀 인디언이 아닌 risc-v 머신코드 bin파일
    import sys
    global MEMORY_START, MEMORY_END, registers, memory, pc
    registers[0] ="0"*32

    if len(sys.argv) < 2:
        print("Usage: python riscv_simulator.py <binary_file> [instr_count] [intr_count]")
        sys.exit(1)

    binary_file = sys.argv[1]  #instr파일

    instr_count = None  #명령어의 개수, 총 명령어 개수를 제한
    intr_count = None #명령의 실행 횟수



    if len(sys.argv) ==3: #인자가 두개면 1번은 instr 파일 2번은 instr 개수
        instr_count = int(sys.argv[2])
        
        #memory load과정
        with open(binary_file, 'rb') as binary_file:
            binary_data = binary_file.read()
            little_endian_data = convert_to_little_endian(binary_data) #리틀인디언으로 바꾸고
            divided_data_list = divide_into_32_bits(little_endian_data) #32비트씩 끊어서
        # print("원래 맞는값",divided_data_list)
        # 메모리 초기화: 모든 위치를 0xFF로 설정
            memory = [0xFF] * (MEMORY_END - MEMORY_START + 1)
            counter=0

            for index,word in reversed(list(enumerate(divided_data_list))):
        #chunk는32비트로 끊어진 상태..
                binary_value_str = ''.join(format(byte, '08b') for byte in word) #word
                #print("binary_value_str:",binary_value_str)
                if intr_count is not None:
                    intr_count+=1
                #counter = 0
                for i in range(0, len(binary_value_str), 8):
                    byte_str = binary_value_str[i:i+8]  # 8자씩 잘라서 이진 문자열 생성
                    memory[counter] = byte_str
                    counter += 1
            # print(memory)







    if len(sys.argv) > 3: #인자가 세개면 1번은 instr 파일 3번은 instr 개수
        instr_count = int(sys.argv[3])
        data_file = sys.argv[2]
        #memory load과정
        with open(binary_file, 'rb') as binary_file:
            binary_data = binary_file.read()

            little_endian_data = convert_to_little_endian(binary_data) #리틀인디언으로 바꾸고
            divided_data_list = divide_into_32_bits(little_endian_data) #32비트씩 끊어서
        # print("원래 맞는값",divided_data_list)

        # 메모리 초기화: 모든 위치를 0xFF로 설정
            memory = [0xFF] * (MEMORY_END - MEMORY_START + 1)
            counter=0
            for index,word in reversed(list(enumerate(divided_data_list))):
        #chunk는32비트로 끊어진 상태..
                binary_value_str = ''.join(format(byte, '08b') for byte in word) #word
                #print("binary_value_str:",binary_value_str)
                if intr_count is not None:
                    intr_count+=1
                #counter = 0
                for i in range(0, len(binary_value_str), 8):
                    byte_str = binary_value_str[i:i+8]  # 8자씩 잘라서 이진 문자열 생성
                    memory[counter] = byte_str
                    counter += 1
            # print(memory)
        with open(data_file, 'rb') as data_binary_file:
            binary_data = data_binary_file.read()
            little_endian_data = convert_to_little_endian(binary_data) #리틀인디언으로 바꾸고
            divided_data_list = divide_into_32_bits(little_endian_data) #32비트씩 끊어서
            data_memeory_counter=0
            for index,word in reversed(list(enumerate(divided_data_list))):
                binary_value_str = ''.join(format(byte, '08b') for byte in word) #word
                #print("binary_value_str 데이터 :",binary_value_str)
                for i in range(0, len(binary_value_str), 8):
                    byte_str = binary_value_str[i:i+8]  # 8자씩 잘라서 이진 문자열 생성
                    data_memory[start_address + data_memeory_counter] = byte_str
                    data_memeory_counter+=1



    for i in range(len(memory)): #초기화
        if memory[i] == 0xFF:
            memory[i] = 0x0
    registers[0] ="0"*32

   # print(instr_count)





#메모리에서 읽어와야행 요소네개씩 읽어와서 리틀인디언으로 바꾸기
   # print("데이터 로드할떄:",memory)
    simulate(instr_count=instr_count, intr_count=intr_count)





riscv_sim_simulater()








