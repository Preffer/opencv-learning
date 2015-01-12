#include <stdexcept>
#include "i18nText.h"

i18nText::i18nText() {
	if(FT_Init_FreeType(&library)){
		throw runtime_error("Failed to initialize FreeType");
	}
}

i18nText::i18nText(const string& path, uint size, float space, float gap) : i18nText() {
	setFont(path);
	setStyle(size, space, gap);
}

i18nText::~i18nText() {
	FT_Done_Face(face);
	FT_Done_FreeType(library);
}

void i18nText::setFont(const string& path) {
	if(FT_New_Face(library, path.c_str(), 0, &face)){
		throw invalid_argument("Failed to load font: " + path);
	}
	FT_Set_Pixel_Sizes(face, size, 0);
}

void i18nText::setStyle(uint size, float space, float gap) {
	this->size = size;
	this->space = space;
	this->gap = gap;
	FT_Set_Pixel_Sizes(face, size, 0);
}

void i18nText::putText(Mat& img, const wstring& text, Point pos, Scalar color) {
	CV_Assert(!img.empty());

	for(const wchar_t& ch : text){
		putWChar(img, ch, pos, color);
	}
}

void i18nText::putWChar(Mat& img, wchar_t wc, Point& pos, Scalar& color) {
	FT_UInt glyph_index = FT_Get_Char_Index(face, wc);
	FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT);
	FT_Render_Glyph(face->glyph, FT_RENDER_MODE_MONO);
	FT_GlyphSlot slot = face->glyph;

	int rows = slot->bitmap.rows;
	int cols = slot->bitmap.width;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int off  = ((0 == 0) ? i : (rows - 1 - i)) * slot->bitmap.pitch + j / 8;
			if (slot->bitmap.buffer[off] & (0xC0 >> (j % 8))) {
				int r = (0 == 0) ? pos.y - (rows - 1 - i) : pos.y + i;
				int c = pos.x + j;

				if (r >= 0 && r < img.rows && c >= 0 && c < img.cols) {
					Vec3b scalar = img.at<Vec3b>(r, c);
					for (int k = 0; k < 3; ++k) {
						scalar.val[k] = color.val[k];
					}
					img.at<Vec3b>(r, c) = scalar;
				}
			}
		}
	}

	pos.x += (int)((cols ? cols : size * space) + size * gap);
}
