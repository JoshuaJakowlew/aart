#ifndef AART_CHARMAP_H
#define AART_CHARMAP_H

#include <string_view>
#include <memory>
#include <algorithm>

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftglyph.h>
#include <freetype/ftcolor.h>
#include <opencv2/opencv.hpp>

#include <aart/ImageManager.hpp>

struct rgb
{
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
};

class Charmap
{
public:
    Charmap(std::string_view filename, int height, const std::string& chars, const std::vector<rgb>& colors) :
        m_chars{std::move(chars)},
        m_colors{std::move(colors)}
    {    
        auto error = FT_New_Face(m_freetype.handle, filename.data(), 0, &m_face);
        if (error == FT_Err_Unknown_File_Format)
        {
            throw std::runtime_error{"Unknown file format"};
        }
        else if (error)
        {
            throw std::runtime_error{"Failed to create face"};
        }

        error = FT_Set_Char_Size(m_face, 0, height * 64, 0, 0);
        if (error)
        {
            throw std::runtime_error{"Failed to set pixel size"};
        }
    }

    auto render() -> cv::Mat
    {
        auto text = renderString(m_chars);
        text.convertTo(text, CV_32FC1, 1./255.);

        auto bg = renderBackground(text.cols, text.rows, m_colors);

        for (int i = 0; i < m_colors.size() * m_colors.size(); ++i)
        {
            cv::Rect roi(0, i * text.rows, text.cols, text.rows);
            auto color = m_colors[i % m_colors.size()];
            auto bg_line = bg(roi);

            blend(text, color.r, color.g, color.b, bg_line);
        }

        m_map = bg;
        m_cellw = text.cols / static_cast<int>(m_chars.length());
        m_cellh = text.rows;
        m_ncells = static_cast<int>(m_chars.length() * m_colors.size() * m_colors.size());
        return m_map;
    }

    [[nodiscard]] auto cellAt(int x, int y) -> cv::Mat
    {
        return m_map({x * m_cellw, y * m_cellh, m_cellw, m_cellh});
    }

    [[nodiscard]] auto charmap() const noexcept -> cv::Mat { return m_map; }
    [[nodiscard]] auto type() const noexcept -> int { return m_map.type(); }
    [[nodiscard]] auto chars() const noexcept { return m_chars; }
    [[nodiscard]] auto colors() const noexcept { return m_colors; }
    [[nodiscard]] auto cellW() const noexcept { return m_cellw; }
    [[nodiscard]] auto cellH() const noexcept { return m_cellh; }
    [[nodiscard]] auto nCells() const noexcept { return m_cellw; }

    ~Charmap()
    {
        FT_Done_Face(m_face);
    }


private:
    struct FreetypeState
    {
        FreetypeState()
        {
            auto error = FT_Init_FreeType(&handle);
            if (error)
            {
                throw std::runtime_error{"Failed to initialize freetype\n"};
            }
        }
    
        ~FreetypeState()
        {
            FT_Done_FreeType(handle);
        }
    
        FT_Library handle;
    };

    static inline FreetypeState m_freetype{};

    FT_Face m_face;
    std::vector<rgb> m_colors;
    std::string m_chars;

    cv::Mat m_map;
    int m_cellw;
    int m_cellh;
    int m_ncells;

    auto blend(const cv::Mat& text, int r, int g, int b, cv::Mat& background) const -> void
    {       
        cv::Mat channels[3];
        cv::split(background, channels);

        float color[] = {b / 255.f, g / 255.f, r / 255.f};

        for (int i = 0; i < 3; ++i)
        {
            cv::multiply(1.f - text, channels[i], channels[i]);
            channels[i] += text * color[i];
        }   

        cv::merge(channels, 3, background);
    }

    [[nodiscard]] auto renderString(std::string_view char_palette) const -> cv::Mat
    {
        FT_GlyphSlot slot = m_face->glyph;

        auto [max_width, max_height] = getMaxCharBBox(char_palette);
        auto [max_bearing_x, max_bearing_y] = getMaxCharBearing(char_palette);
        // BUG: Line height is broken for negative y bearing
        cv::Mat atlas(std::max(max_height, max_bearing_y), max_width * static_cast<int>(char_palette.length()), CV_8U, {0, 0, 0, 0});

        for (int i = 0; i < char_palette.length(); ++i)
        {
            auto error = FT_Load_Char(m_face, char_palette[i], FT_LOAD_RENDER);
            int bitmap_width = slot->bitmap.width;
            int bitmap_height = slot->bitmap.rows;
            int bearing_x = slot->metrics.horiBearingX / 64;
            int bearing_y = slot->metrics.horiBearingY / 64;

            cv::Mat cell(bitmap_height, bitmap_width, CV_8U, slot->bitmap.buffer);
            auto matRoi = atlas({i * max_width + bearing_x, 0 + max_bearing_y - bearing_y, bitmap_width, bitmap_height});
            cell.copyTo(matRoi);
        }

        cv::extractChannel(atlas, atlas, 0);
        return atlas;
    }

    [[nodiscard]] auto renderBackground(unsigned line_width, unsigned line_height, const std::vector<rgb>& colors) const -> cv::Mat
    {
        cv::Mat bg(static_cast<int>(line_height * colors.size() * colors.size()), line_width, CV_32FC3);
        for (int i = 0; i < colors.size(); ++i)
        {
            cv::Rect roi(0, i * line_height * static_cast<int>(colors.size()), line_width, static_cast<int>(line_height * colors.size()));
            cv::Scalar color{colors[i].b / 255.f, colors[i].g / 255.f, colors[i].r / 255.f};
            bg(roi) = color;
        }

        return bg;
    }

    [[nodiscard]] auto getCharBBox(char c) const -> std::tuple<int, int>
    {
        FT_GlyphSlot slot = m_face->glyph;
        auto error = FT_Load_Char(m_face, c, FT_LOAD_RENDER);
        // TODO: Handle error
        return std::make_tuple(slot->advance.x / 64, slot->metrics.height / 64);
    }

    [[nodiscard]] auto getCharBearing(char c) const -> std::tuple<int, int>
    {
        FT_GlyphSlot slot = m_face->glyph;
        auto error = FT_Load_Char(m_face, c, FT_LOAD_RENDER);
        // TODO: Handle error
        return std::make_tuple(slot->metrics.horiBearingX / 64, slot->metrics.horiBearingY / 64);
    }

    [[nodiscard]] auto getMaxCharBBox(std::string_view char_palette) const -> std::tuple<int, int>
    {
        FT_GlyphSlot slot = m_face->glyph;

        int max_width = 0;
        int max_height = 0;
        for (auto c : char_palette)
        {
            auto [char_width, char_height] = getCharBBox(c);
            max_width = std::max(max_width, char_width);
            max_height = std::max(max_height, char_height);
        }

        return std::make_tuple(max_width, max_height);
    }

    [[nodiscard]] auto getMaxCharBearing(std::string_view char_palette) const -> std::tuple<int, int>
    {
        FT_GlyphSlot slot = m_face->glyph;

        int max_bearing_x = 0;
        int max_bearing_y = 0;
        for (auto c : char_palette)
        {
            auto [char_bearing_x, char_bearing_y] = getCharBearing(c);
            max_bearing_x = std::max(max_bearing_x, char_bearing_x);
            max_bearing_y = std::max(max_bearing_y, char_bearing_y);
        }

        return std::make_tuple(max_bearing_x, max_bearing_y);
    }
};

#endif
