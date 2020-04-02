from PIL import Image, ImageDraw, ImageFont, ImageColor

# This is ascii grayscale line starting from bright to dark
# Currently aart supports only this line
ascii = ' .:-=+*#%@'

# This is color palette. This palette is used as default and is created from
# cmd.exe 16-color default palette.
# You can increase or decrease number of colors
# Any color is a 3-tuple of red, green and blue channels
# Example: (255, 0, 0) is red color

palette = [
(33, 122, 216),
(30, 36, 86),
(42, 91, 246),
(43, 90, 254),
(53, 210, 251),
(23, 42, 181),
(26, 70, 205),
(22, 195, 227),
(31, 251, 251),
(241, 229, 251),
(54, 244, 255),
(71, 130, 220),
(250, 227, 235),
(50, 78, 161),
(25, 40, 195),
(24, 153, 208),
(91, 171, 232),
(59, 236, 244),
(72, 160, 234),
(33, 63, 223),
(7, 67, 191),
(64, 37, 56),
(43, 118, 160),
(132, 38, 122),
(0, 69, 159),
(130, 35, 117),
(209, 45, 220),
(30, 104, 213),
(220, 46, 229),
(65, 210, 237),
(130, 37, 118),
(48, 243, 249),
]

#palette = list(map(ImageColor.getrgb, palette))

palette_len = len(palette)

colormap = Image.new('RGB', (palette_len, 1))
draw = ImageDraw.Draw(colormap)

for x, color in enumerate(palette):
    draw.point((x, 0), fill=color)

colormap.save('colormap.png')

# You can use any TrueType font. I use Courier New.
font = ImageFont.truetype('Courier New.ttf', 12)

img = Image.new('RGB', (1, 1))
draw = ImageDraw.Draw(img)

line_size = draw.textsize(ascii, font, spacing=0)
line_size = (line_size[0] - len(ascii), line_size[1]) # I don't know who is wrong, me or Pillow

perm_n = palette_len * palette_len
charmap_size = (line_size[0], line_size[1] * perm_n)
charmap = Image.new('RGB', charmap_size)
draw = ImageDraw.Draw(charmap)

for i, bg in enumerate(palette):
    for j, fg in enumerate(palette):
        offset_y = line_size[1] * (i * palette_len + j)
        start_pos = (0, offset_y)
        end_pos = (line_size[0], offset_y + line_size[1])

        draw.rectangle([start_pos, end_pos], fill=bg)
        draw.text(start_pos, ascii, fill=fg, spacing=0)

charmap.save('charmap.png')
