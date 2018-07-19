# coding:utf-8
from reportlab.pdfgen import canvas


def hello(program_name, current_time, time_consumption, loophole, taint):
    c = canvas.Canvas("helloworld.pdf")
    c.setFont("Helvetica-Bold", 15)
    c.drawCentredString(297.635, 800, "Vulnerability Detection Report", )
    c.setFont("Times-Italic", 10)
    c.drawCentredString(297.635, 770, "Time: " + current_time + "    Time consumption: " + time_consumption
                        + "    NumOfLoophole: " + str(len(loophole))
                        + "    NumOfTaints: " + str(len(taint)))
    c.setFont("Courier-Bold",12)
    c.drawString(60, 740, "Program Name: ")
    c.setFont("Helvetica-Oblique", 11)
    c.drawString(160, 740, program_name)
    c.setFont("Courier-Bold", 12)
    c.drawString(60, 710, "Loophole Details:")
    line = 710
    c.setFont("Helvetica-Oblique", 11)
    # 循环输出所有的漏洞相关信息
    for i in range(len(loophole)):
        if line < 30:
            c.showPage()
            line = 800
        c.drawString(80, line - 30, str(i + 1) + " Type: " + loophole[i][1]
                     + "    Coordinate line: " + str(loophole[i][0])
                     + "    Taint: taint")

        line -= 30
        c.drawString(80, line - 30, "   Line content: lines")
        line -= 30

    # c.drawString(80, )
    c.setFont("Courier-Bold", 12)
    c.drawString(60, line - 30, "Taints(suspicious): ")
    c.drawString(160, line - 30, "    ".join([k[1] for k in taint]))
    line -= 30
    # 循环输出所有污点相关信息
    j = 1
    c.setFont("Helvetica-Oblique", 11)
    for i in range(len(taint)):
        if line < 30:
            c.showPage()
            line = 800
        if taint[i] in [k[0] for k in loophole]:
            continue
        c.drawString(80, line - 30, str(j)
                     + "    Name: " + taint[i][1]
                     + "    Coordinate line: " + str(taint[i][0]))

        line -= 30
        c.drawString(80, line - 30, "   Line content: lines")
        line -= 30
        j += 1
    # for i in range(len(taint))
    c.showPage()
    # c.drawString(10, 830, "hhhh")
    # c.showPage()
    c.save()


hello("http：jfsdjf", "fsdfsd", "fsdfsdf", [[10, "h"], [12, "s"]], [[1,"ab"],[2,"fsdf"]])
