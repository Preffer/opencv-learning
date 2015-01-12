#ifndef _I18N_TEXT_H_
#define _I18N_TEXT_H_

#include <opencv2/highgui/highgui.hpp>
#include <ft2build.h>
#include FT_FREETYPE_H

using namespace std;
using namespace cv;

class i18nText {
public:
	i18nText();
	i18nText(const string& path, uint size = 25, float space = 0.5, float gap = 0.1);
	~i18nText();

	void setFont(const string& path);
	void setStyle(uint size, float space, float gap);
	void putText(Mat& img, const wstring& text, Point pos, Scalar color = CV_RGB(0, 0, 0));

private:
	void putWChar(Mat& img, wchar_t wc, Point& pos, Scalar& color);

	FT_Library library;
	FT_Face face;
	uint size = 25;
	float space = 0.5;
	float gap = 0.1;
};

#endif // _I18N_TEXT_H_
