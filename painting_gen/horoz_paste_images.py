import sys
from PIL import Image

monat_name = sys.argv[1]
photo_name = sys.argv[2]
paint_name = sys.argv[3]
output_name = sys.argv[4]

images = [Image.open(x) for x in [monat_name, photo_name, paint_name]]
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save(output_name)
