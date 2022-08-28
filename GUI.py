
import dearpygui.dearpygui as dpg
from char_detection import char_detection_recognition
from torchvision import models
import pandas as pd
dpg.create_context()


def get_width_and_hight(width, height):
    w = 500
    h = 650
    if width < w:
        w = width
    if height < h:
        h = height
    return w, h


selected_imge_path = None


def load_img(sender, app_data):
    global selected_imge_path
    print("Sender: ", sender)
    print("App Data: ", app_data["file_path_name"])
    selected_imge_path = app_data["file_path_name"]
    width, height, channels, data = dpg.load_image(app_data["file_path_name"])

    with dpg.texture_registry(show=False):
        dpg.add_static_texture(width=width, height=height,
                               default_value=data, tag="texture_tag")

    x, y = get_width_and_hight(width=width, height=height)
    dpg.add_image("texture_tag", parent=w3)
    dpg.set_item_height(w3, height=y)
    dpg.set_item_width(w3, width=x)
    w3_x, w3_y = dpg.get_item_pos(w3)
    dpg.set_item_pos(w3, (w3_x, w3_y+37))
    dpg.show_item(w3)
    xx, yy = dpg.get_item_pos("button1")
    dpg.set_item_pos("button2", (x + xx + 25, yy))

    dpg.show_item("button2")
    dpg.add_window(parent=w2, tag="new", pos=(
        x + xx + 25, yy+25), height=y, width=x, horizontal_scrollbar=True)


def show_char():
    char_image_path = "C:\\Users\\parvi\\Desktop\\Project\\test_image"
    image_path = selected_imge_path
    Model_classes_Info = "training_data.csv"
    model = models.resnet18()
    model_path = './generated/resnet_25.pth'
    output_class_number = 212
    dpg.add_text(default_value='SetUp Compelited,', parent="new")
    c_d_r = char_detection_recognition(
        image_path, Model_classes_Info, model, model_path, output_class_number)
    dpg.add_text(default_value='Finding characters', parent="new")
    c_d_r.image_to_bboxs()
    dpg.add_text(default_value='cropping characters', parent="new")
    c_d_r.extract_test_chars_from_bboxs()
    dpg.add_text(default_value='Predicting classes', parent="new")
    data_df, tes_label = c_d_r.predic()
    print(data_df.head())
    dpg.add_text(default_value='loading the table', parent="new")
    # data_df = pd.read_csv(
    #    "predictions.csv", encoding="utf-8", squeeze=True)
    l = data_df.values.tolist()
    chars = []
    labels = []
    # print(tes_label)
    for s in l:
        c = s.split()
        chars.append(c[1])
    for s in tes_label:
        c = str(s).split()
        labels.append(c[1])
    # print(chars)
    num = len(chars)
    with dpg.texture_registry(show=False):
        for i in range(1, num):
            img_path = char_image_path + "\\" + str(i) + ".jpg"
            width, height, channels, data = dpg.load_image(img_path)
            dpg.add_static_texture(width=width, height=height,
                                   default_value=data, tag="texture_tag"+str(i))
    with dpg.table(parent="new", borders_outerH=True, borders_innerH=True):
        # use add_table_column to add columns to the table,
        # table columns use child slot 0,

        dpg.add_table_column(label="Char")
        dpg.add_table_column(label="my")
        dpg.add_table_column(label="Other")

        dpg.add_table_column(label="Character")
        dpg.add_table_column(label="my predction")
        dpg.add_table_column(label="Tess prediction")

        dpg.add_table_column(label="Character")
        dpg.add_table_column(label="my predction")
        dpg.add_table_column(label="Tess prediction")

        dpg.add_table_column(label="Character")
        dpg.add_table_column(label="my predction")
        dpg.add_table_column(label="Tess prediction")

        dpg.add_table_column(label="Character")
        dpg.add_table_column(label="my predction")
        dpg.add_table_column(label="Tess prediction")
        # add_table_next_column will jump to the next row
        # once it reaches the end of the columns
        # table next column use slot 1

        for i in range(0, (num//10)-1):
            with dpg.table_row():
                for j in range(1, 16):
                    if j % 3 == 1:
                        dpg.add_image("texture_tag"+str(i*5+j//3+1),
                                      )
                    elif j % 3 == 2:
                        dpg.add_text(chars[i*5+j//3])
                    else:
                        dpg.add_text(labels[i*5+j//3-1][0])

    # dpg.show_item("new")
# def changeWindow():
#    dpg.set_primary_window(w2, True)
#    dpg.configure_item(w2, show=True)


with dpg.theme() as first_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg,
                            (238, 232, 170), category=dpg.mvThemeCat_Core)
#        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,
#                            5, category=dpg.mvThemeCat_Core)

with dpg.font_registry():
    with dpg.font("GentiumBookPlus-Bold.ttf", 20) as font1:
        dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
        dpg.add_font_range(0x1F00, 0x1FFF)
        dpg.add_font_range(0x0370, 0x03FF)


with dpg.window(tag="First Window", width=500, height=300, autosize=False, horizontal_scrollbar=False) as w1:

    with dpg.file_dialog(directory_selector=False, show=False, callback=load_img, tag="file_dialog_tag"):
        dpg.add_file_extension(".jpg", color=(255, 0, 255, 255))
        dpg.add_file_extension(".png", color=(255, 0, 255, 255))
        dpg.add_file_extension(".*")

    with dpg.group(horizontal=True):
        dpg.add_button(label="Select Your Image", tag="button1",
                       callback=lambda: dpg.show_item("file_dialog_tag"))

        dpg.add_button(label="Show Text", tag="button2",
                       show=False, callback=show_char)
    with dpg.group(horizontal=True) as w2:
        with dpg.window(label="Tutorial", horizontal_scrollbar=True, show=False) as w3:
            pass

# with dpg.window(tag="Second  Window", width=500, height=300, show=False) as w2:
#    dpg.add_text(":)))", tag="text item 2")
#dpg.bind_item_theme(w1, first_theme)

dpg.bind_font(font1)
# dpg.show_style_editor()

dpg.create_viewport(title='Custom Title')
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("First Window", True)
dpg.start_dearpygui()
dpg.destroy_context()

"""
number of errors: 4
error rate : 0.08
Training complete in 23m 50s

"""
