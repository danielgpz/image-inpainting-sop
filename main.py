import os
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from ImageInpainting import CorruptedImage, read_image_as_mask

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)

class RandomMaskDialog(FloatLayout):
    corrupt = ObjectProperty(None)
    corrupt_prob = ObjectProperty(None)
    cancel = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)

class InpaintingDialog(FloatLayout):
    inpainting = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Controller(Widget):
    # Subwidget properties
    image_screen = ObjectProperty(None)
    load_image_bt = ObjectProperty(None)
    load_mask_bt = ObjectProperty(None)
    random_mask_bt = ObjectProperty(None)
    inpainting_image_bt = ObjectProperty(None)
    # Load & save properties
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    
    
    def __init__(self, **kwargs):
        super(Controller, self).__init__(**kwargs)
        self.LoadDialog = LoadDialog
        self.RandomMaskDialog = RandomMaskDialog
        self.SaveDialog = SaveDialog
        self.InpaintingDialog = InpaintingDialog
        
        self.image_dir = ''
        self.temp_dir = ''
        self.corrupted_image = None

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_popup(self, title, klass, size_hint, **kwargs):
        content = klass(**kwargs)
        self._popup = Popup(title=title, content=content, size_hint=size_hint)
        self._popup.open()

    def load_image(self, path, filename):
        file = os.path.join(path, filename[0])
        self.image_screen.source = file
        self.image_dir = file
        self.corrupted_image = None
        
        self.load_mask_bt.disabled = False
        self.random_mask_bt.disabled = False
        self.inpainting_image_bt.disabled = True
        self.dismiss_popup()

    def load_mask(self, path, filename):
        file = os.path.join(path, filename[0])
        mask = read_image_as_mask(file)
        
        self.corrupted_image = CorruptedImage(self.image_dir, mask=mask)

        if not self.temp_dir:
            fmt = self.image_dir.rsplit('.', 1)[1]
            self.temp_dir = f'./temp/{hash(self)}.{fmt}'
            self.image_screen.source = self.temp_dir
        
        self.corrupted_image.save(self.temp_dir)
        self.image_screen.reload()
        
        self.inpainting_image_bt.disabled = False
        self.dismiss_popup()

    def corrupt_image(self, corrupt_prob):
        self.corrupted_image = CorruptedImage(self.image_dir, corrupt_prob=corrupt_prob)

        if not self.temp_dir:
            fmt = self.image_dir.rsplit('.', 1)[1]
            self.temp_dir = f'./temp/{hash(self)}.{fmt}'
            self.image_screen.source = self.temp_dir
        
        self.corrupted_image.save(self.temp_dir)
        self.image_screen.reload()

        self.inpainting_image_bt.disabled = False
        self.dismiss_popup()

    def inpainting(self, **kwargs):
        self._popup.disabled = True
        self.corrupted_image.inpainting(**kwargs)
        
        self.corrupted_image.save(self.temp_dir)
        self.image_screen.reload()
        
        self.dismiss_popup()

    def save_image(self, path, filename):
        file = os.path.join(path, filename[0])
        print(f'Saving file {file}')

        self.dismiss_popup()


class SopApp(App):
    def build(self):
        return = Controller()


if __name__ == '__main__':
    SopApp().run()