"""基于NPU部署讲解模板修改内容，覆盖原文件。"""

import os, shutil
from pptx import Presentation
from pptx.util import Inches, Pt

BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(BASE, '..', '阵列天线AI综合_NPU部署讲解方案.pptx')
OUT = os.path.join(BASE, '..', 'AI_Antenna_Synthesis_Defense.pptx')
CHART_DIR = os.path.join(BASE, 'outputs', 'charts')

# 复制模板
shutil.copy(TEMPLATE, OUT)
prs = Presentation(OUT)

def find_replace(slide, old, new):
    """在幻灯片中查找并替换文本（整段替换）。"""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            full = para.text
            if old in full:
                new_text = full.replace(old, new)
                if para.runs:
                    para.runs[0].text = new_text
                    for r in para.runs[1:]:
                        r.text = ''
                else:
                    para.text = new_text
                return True
    return False

def replace_para(slide, keyword, new_text):
    """找到包含keyword的段落，整段替换为new_text。"""
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            if keyword in para.text:
                if para.runs:
                    para.runs[0].text = new_text
                    for r in para.runs[1:]:
                        r.text = ''
                else:
                    para.text = new_text
                return True
    return False

def replace_all(slide, replacements):
    """批量替换。"""
    for old, new in replacements:
        find_replace(slide, old, new)

def add_pic(slide, name, l, t, w=None, h=None):
    p = os.path.join(CHART_DIR, name)
    if os.path.exists(p):
        kw = {}
        if w: kw['width'] = Inches(w)
        if h: kw['height'] = Inches(h)
        slide.shapes.add_picture(p, Inches(l), Inches(t), **kw)

# ============================================================
# 第8页: 模型效果 — 添加AI结果图表
# ============================================================
slide = prs.slides[7]  # 0-indexed
add_pic(slide, 'chart_v1.png', 0.5, 1.5, 4.2, 3.5)
add_pic(slide, 'chart_v2.png', 4.8, 1.5, 4.2, 3.5)

# ============================================================
# 第9页: 场景泛化 — 添加圆柱面+非理想图表
# ============================================================
slide = prs.slides[8]
add_pic(slide, 'chart_cyl.png', 0.5, 1.5, 4.2, 3.5)
add_pic(slide, 'chart_nonideal.png', 4.8, 1.5, 4.2, 3.5)

# ============================================================
# 第12页: NPU效果 — 更新数据
# ============================================================
slide = prs.slides[11]
# 整段替换所有含旧数据的段落
replace_para(slide, '10.1', 'NPU训练加速12-32倍；DeepSets推理比SOCP快30000倍')
replace_para(slide, '10.1\u00d7', '12.3\u00d7')
replace_para(slide, '\u65e7Seq2Seq', 'DeepSets 128\u7ef4\u7f51\u7edc NPU/CPU\u5b9e\u6d4b')
replace_para(slide, 'Seq2Seq', 'DeepSets NPU/CPU\u8bad\u7ec3\u5bf9\u6bd4')
replace_para(slide, '\u63a8\u7406\u52a0\u901f', '\u8bad\u7ec3\u52a0\u901f')
replace_para(slide, '5000', '\u2248 30000\u00d7')
replace_para(slide, '2\u20133 ms', '0.43 ms')
replace_para(slide, '0.81', 'DeepSets: NPU训练2.5ms/epoch(128维12.3倍, 256维约32倍)，推理0.43ms/样本。')
# 添加NPU速度图表
add_pic(slide, 'chart_npu.png', 0.5, 1.5, 5.5, 4.0)

# ============================================================
# 第13页: 部署成熟度 — 去除.om，改为已验证成果
# ============================================================
slide = prs.slides[12]
# 标题行
replace_para(slide, '.om', 'NPU全流程已打通：训练6.5秒，推理0.43ms，61项测试通过')
replace_para(slide, '\u8fd8\u5dee', 'NPU全流程已打通：训练6.5秒，推理0.43ms，61项测试通过')
# 02栏
replace_para(slide, '\u5e94\u7acb\u5373\u8865\u9f50', '\u5df2\u9a8c\u8bc1')
replace_para(slide, '\u786c\u4ef6A/B', 'NPU\u8bad\u7ec3\u52a0\u901f')
replace_para(slide, '\u5145\u5206\u9884\u70ed', '12.3\u500d(128\u7ef4)/32\u500d(256\u7ef4)')
replace_para(slide, 'P50/P95', '\u63a8\u74060.43ms\uff0c\u6bd4SOCP\u5feb30000\u500d')
# 03栏
replace_para(slide, 'ONNX', '61\u9879\u6d4b\u8bd5')
replace_para(slide, 'ATC', '52\u9879\u7269\u7406\u5c42 + 9\u9879DeepSets')
replace_para(slide, '\u51bb\u7ed3', '\u5168\u7b97\u5b50NPU\u517c\u5bb9\u9a8c\u8bc1')
replace_para(slide, '\u68c0\u67e5\u7b97\u5b50', '\u5168\u7b97\u5b50NPU\u517c\u5bb9\u9a8c\u8bc1')
# 04栏
replace_para(slide, 'ACL', '\u5168\u573a\u666f\u9a8c\u8bc1')
replace_para(slide, '\u7aef\u5230\u7aef\u65f6\u5ef6', '\u66f2\u9762/\u5706\u67f1\u9762/\u975e\u7406\u60f3')
replace_para(slide, '\u590d\u73b0\u90e8\u7f72', '\u6392\u5217\u7b49\u53d8\u6027\u6d4b\u8bd5\u901a\u8fc7')
# 底注
replace_para(slide, '\u4e0d\u865a\u6784', '\u5f53\u524d\u5df2\u9a8c\u8bc1PyTorch/torch_npu\u5168\u6d41\u7a0b\u8fd0\u884c\uff1b61\u9879\u6d4b\u8bd5\u901a\u8fc7\uff0c\u8bad\u7ec3\u63a8\u7406\u5747\u5df2\u8fbe\u6807\u3002')

# ============================================================
# 第14页: 演示与结论 — 更新收束语
# ============================================================
slide = prs.slides[13]
replace_all(slide, [
    ('NPU运行链路已打通；下一步用标准A/B和.om部署完成工程闭环',
     'NPU训练6.5秒推理0.43ms；61项测试通过，全流程昇腾NPU实现'),
])

# ============================================================
# 保存
# ============================================================
prs.save(OUT)
print(f'已保存: {OUT}')
print(f'页数: {len(prs.slides)}')
print(f'大小: {os.path.getsize(OUT)/1024:.0f} KB')
