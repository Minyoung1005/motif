from PIL import Image

def concatenate_images(images, max_rows=None, max_cols=None, margin=2, bg_color=(255, 255, 255), save_path=None, label=None):
    if max_cols == 1:
        width = max(img.width for img in images)
        total_height = sum(img.height for img in images) + margin * (len(images) - 1)

        new_img = Image.new('RGB', (width, total_height), bg_color)

        current_height = 0
        for img in images:
            new_img.paste(img, (0, current_height))
            current_height += img.height + margin  # adding a 20px black bar
        
    elif max_rows == 1:
        height = max(img.height for img in images)
        total_width = sum(img.width for img in images) + margin * (len(images) - 1)

        new_img = Image.new('RGB', (total_width, height), bg_color)

        current_width = 0
        for img in images:
            new_img.paste(img, (current_width, 0))
            current_width += img.width + margin
    else:
        print("Not implemented yet")
    
    # save image if save_path is provided
    if save_path is not None:
        new_img.save(save_path)

    return new_img