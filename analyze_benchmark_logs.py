import re
import sys

def parse_log(log_path):
    print(f"\n=======================================================")
    print(f"Analyzing: {log_path}")
    print(f"=======================================================")
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    blocks = content.split("========================================")
    
    stats = {
        'add_le3': {'correct': 0, 'total': 0},
        'add_4digit': {'correct': 0, 'total': 0},
        'sub_le3': {'correct': 0, 'total': 0},
        'sub_4digit': {'correct': 0, 'total': 0},
        'mul_all': {'correct': 0, 'total': 0},
        'div_single_divisor': {'correct': 0, 'total': 0},
        'div_multi_divisor': {'correct': 0, 'total': 0},
        'div_all': {'correct': 0, 'total': 0},
        'overall': {'correct': 0, 'total': 0}
    }
    
    for block in blocks:
        if not block.strip():
            continue
            
        m_correct = re.search(r'\(Correct:\s*(True|False)\)', block)
        if not m_correct:
            continue
        is_correct = (m_correct.group(1) == 'True')
        
        m_q = re.search(r'Q:\s*What is\s*(\d+)\s*([\+\-\*\/])\s*(\d+)\?', block)
        if not m_q:
            continue
            
        a = int(m_q.group(1))
        op = m_q.group(2)
        b = int(m_q.group(3))
        
        stats['overall']['total'] += 1
        if is_correct:
            stats['overall']['correct'] += 1
            
        if op == '+':
            if a < 1000 and b < 1000:
                stats['add_le3']['total'] += 1
                if is_correct: stats['add_le3']['correct'] += 1
            else:
                stats['add_4digit']['total'] += 1
                if is_correct: stats['add_4digit']['correct'] += 1
        elif op == '-':
            if a < 1000 and b < 1000:
                stats['sub_le3']['total'] += 1
                if is_correct: stats['sub_le3']['correct'] += 1
            else:
                stats['sub_4digit']['total'] += 1
                if is_correct: stats['sub_4digit']['correct'] += 1
        elif op == '*':
            stats['mul_all']['total'] += 1
            if is_correct: stats['mul_all']['correct'] += 1
        elif op == '/':
            stats['div_all']['total'] += 1
            if is_correct: stats['div_all']['correct'] += 1
            if b < 10:
                stats['div_single_divisor']['total'] += 1
                if is_correct: stats['div_single_divisor']['correct'] += 1
            else:
                stats['div_multi_divisor']['total'] += 1
                if is_correct: stats['div_multi_divisor']['correct'] += 1

    def fmt(d):
        c, t = d['correct'], d['total']
        pct = (c / t * 100) if t > 0 else 0
        return f"{c}/{t} ({pct:.2f}%)"

    print(f"Addition (<=3 digits, In-Dist):   {fmt(stats['add_le3'])}")
    print(f"Addition (4 digits, Out-of-Dist):  {fmt(stats['add_4digit'])}")
    print(f"Subtraction (<=3 digits, In-Dist):{fmt(stats['sub_le3'])}")
    print(f"Subtraction (4 digits, Out-of-Dist): {fmt(stats['sub_4digit'])}")
    print(f"Multiplication (All):              {fmt(stats['mul_all'])}")
    print(f"Division (Single-digit divisor):   {fmt(stats['div_single_divisor'])}")
    print(f"Division (Multi-digit divisor):    {fmt(stats['div_multi_divisor'])}")
    print(f"Division (All):                    {fmt(stats['div_all'])}")
    print(f"-------------------------------------------------------")
    print(f"Overall Benchmark Accuracy:        {fmt(stats['overall'])}")

if __name__ == '__main__':
    for path in sys.argv[1:]:
        parse_log(path)
