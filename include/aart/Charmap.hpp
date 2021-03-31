#ifndef AART_CHARMAP_H
#define AART_CHARMAP_H

#include <string_view>
#include <memory>

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftglyph.h>
#include <opencv2/opencv.hpp>

#include <aart/utility.hpp>
struct rgb
{
    unsigned char r = 0;
    unsigned char g = 0;
    unsigned char b = 0;
};

using FontSize = fluent::NamedType<unsigned, struct FontSizeTag, fluent::Callable, fluent::Printable>;
using CharPalette = fluent::NamedType<std::string, struct CharPaletteTag, fluent::Callable, fluent::Printable>;
using CharPaletteView = fluent::NamedType<std::string_view, struct CharPaletteViewTag, fluent::Callable, fluent::Printable>;
using ColorPalette = fluent::NamedType<std::vector<rgb>, struct ColorPaletteTag, fluent::Callable>;

using BearingX = fluent::NamedType<int, struct BearingXTag, fluent::Callable, fluent::Printable>;
using BearingY = fluent::NamedType<int, struct BearingYTag, fluent::Callable, fluent::Printable>;

using VerticalResolution = fluent::NamedType<FT_UInt, struct VerticalResolutionTag, fluent::Callable, fluent::Printable>;
using HorizontalResolution = fluent::NamedType<FT_UInt, struct HorizontalResolutionTag, fluent::Callable, fluent::Printable>;

class Charmap final
{
public:
    Charmap(FilenameView filename, FontSize height, CharPalette const& chars, ColorPalette const& colors) :
        m_chars{std::move(chars)},
        m_colors{std::move(colors)}
    {    
        if(FT_Err_Ok != loadFace(filename, height))
        {
            throw std::runtime_error{"Unable to load face"};
        }
    }

    auto render() -> Matrix<float, 3> &
    {
        auto text = renderString(CharPaletteView{m_chars.get()});
        // Convert to normalized [0..1] single channel float matrix
        auto normalized_text = Matrix<float, 1>{};
        text->convertTo(normalized_text.get(), CV_32FC1, 1./255.);

        const auto text_w = Width{normalized_text->cols};
        const auto text_h = Height{normalized_text->rows};
        auto bg = renderBackground(text_w, text_h, m_colors);

        const auto ncolors = Length{static_cast<int>(m_colors->size())};
        for (int i = 0; i < ncolors * ncolors; ++i)
        {
            const auto roi = RegionOfInterest{{
                PosX{0},
                PosY{i * text_h},
                Width{text_w},
                Height{text_h}
            }};
            const auto color = m_colors.get()[i % ncolors];
            auto bg_line = Matrix<float, 3>{bg.get()(roi)};
            blend(normalized_text, color, bg_line);
        }

        m_map = std::move(bg);
        return m_map;
    }

    [[nodiscard]] auto cellAt(PosX x, PosY y) -> Matrix<float, 3>
    {
        return Matrix<float, 3>{
            m_map.get()({
                PosX{x * m_cellw},
                PosY{y * m_cellh},
                Width{m_cellw},
                Height{m_cellh}
            })
        };
    }

    [[nodiscard]] auto ratio() const noexcept { return static_cast<double>(m_cellw) / m_cellh; }

    [[nodiscard]] auto charmap() const noexcept { return m_map; }
    [[nodiscard]] auto type() const noexcept { return m_map->type(); }
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
                throw std::runtime_error{"Failed to initialize freetype"};
            }
        }
    
        ~FreetypeState()
        {
            FT_Done_FreeType(handle);
        }
    
        FT_Library handle;
    };

    struct StringMetrics
    {
        Width max_width;
        Height max_height;
        BearingX max_bearing_x;
        BearingY max_bearing_y;
        Height line_height;
        Width line_width;
    };

    struct CharMetrics
    {
        Width width;
        Height height;
        BearingX bearing_x;
        BearingY bearing_y;
    };

    static inline FreetypeState m_freetype{};

    FT_Face m_face;
    ColorPalette m_colors;
    CharPalette m_chars;

    Matrix<float, 3> m_map;
    Width m_cellw;
    Height m_cellh;
    Length m_ncells;

    auto blend(Matrix<float, 1> & text, rgb blend_color, Matrix<float, 3> & background) const -> void
    {       
        cv::Mat channels[3];
        cv::split(background, channels);

        float color[] = {
            blend_color.b / 255.f,
            blend_color.g / 255.f,
            blend_color.r / 255.f
        };

        for (int i = 0; i < 3; ++i)
        {
            cv::multiply(1.f - text, channels[i], channels[i]);
            channels[i] += text * color[i];
        }   

        cv::merge(channels, 3, background.get());
    }

    [[nodiscard]] auto renderString(CharPaletteView char_palette) -> Matrix<unsigned char, 1>
    {
        FT_GlyphSlot slot = m_face->glyph;

        const auto metrics = getStringMetrics(char_palette);
        auto atlas = Matrix<unsigned char, 3>{{
            metrics.line_height,
            metrics.line_width,
            CV_8U,
            cv::Scalar{0, 0, 0, 0}
        }};

        for (int i = 0; i < char_palette->size(); ++i)
        {
            const auto [width, height, bearing_x, bearing_y] = getCharMetrics(Index{char_palette.get()[i]});

            const auto pos_x = PosX{i * metrics.max_width};
            const int pos_y = PosY{metrics.max_bearing_y - bearing_y};

            const auto roi = RegionOfInterest{{pos_x, pos_y, width, height}};
            auto cell_roi = atlas.get()(roi);
            auto cell = Submatrix<unsigned char, 3>{{height, width, CV_8U, slot->bitmap.buffer}};
            cell->copyTo(cell_roi);
        }

        m_cellw = Width{metrics.max_width};
        m_cellh = Height{metrics.line_height};
        m_ncells = Length{static_cast<int>(char_palette->size())};

        auto gray_atlas = Matrix<unsigned char, 1>{};
        cv::extractChannel(atlas.get(), gray_atlas.get(), 0);
        return gray_atlas;
    }

    [[nodiscard]] auto renderBackground(Width line_width, Height line_height, ColorPalette const & colors) const -> Matrix<float, 3>
    {
        const auto ncolors = static_cast<int>(colors->size());
        const auto bg_height = Height{line_height * ncolors * ncolors};
        auto bg = Matrix<float, 3>{{
            bg_height,
            line_width,
            CV_32FC3
        }};
        for (auto i = 0; i < ncolors; ++i)
        {
            const auto roi = RegionOfInterest({
                PosX{0},
                PosY{i * line_height * ncolors},
                Width{line_width},
                Height{line_height * ncolors}
            });
            cv::Scalar color{
                colors.get()[i].b / 255.f,
                colors.get()[i].g / 255.f,
                colors.get()[i].r / 255.f
            };
            bg.get()(roi) = color;
        }

        return bg;
    }

    [[nodiscard]] auto getStringMetrics(CharPaletteView char_palette) const -> StringMetrics
    {
        auto max_width = Width{0};
        auto max_height = Height{0};
        auto max_bearing_x = BearingX{0};
        auto max_bearing_y = BearingY{0};
        auto max_underline_height = Height{0};
        
        auto min_bearing_y = BearingY{0}; 
        FT_GlyphSlot slot = m_face->glyph;
        for (auto c : char_palette.get())
        {
            auto error = FT_Load_Char(
                m_face,
                Index_<FT_ULong>{static_cast<FT_ULong>(c)},
                FT_LOAD_RENDER
            );
            // TODO: Handle error
            
            const auto width = Width{static_cast<int>(slot->bitmap.width)};
            const auto height = Height{static_cast<int>(slot->bitmap.rows)};
            const auto bearing_x = BearingX{slot->metrics.horiBearingX >> 6}; // 26.6 fixed point
            const auto bearing_y = BearingY{slot->metrics.horiBearingY >> 6}; // 26.6 fixed point
            
            auto underline_height = Height{0};
            if (bearing_y < BearingY{0})
                underline_height = height;
            else if (bearing_y < height)
                underline_height = Height{height.get() - bearing_y.get()};

            max_width            = std::max(max_width           , width           );
            max_height           = std::max(max_height          , height          );
            max_bearing_x        = std::max(max_bearing_x       , bearing_x       );
            max_bearing_y        = std::max(max_bearing_y       , bearing_y       );
            max_underline_height = std::max(max_underline_height, underline_height);
            
            min_bearing_y        = std::min(min_bearing_y       , bearing_y       );
        }

        return {
            .max_width     = max_width,
            .max_height    = max_height,
            .max_bearing_x = max_bearing_x,
            .max_bearing_y = max_bearing_y,
            .line_height   = Height{max_underline_height + (max_bearing_y - min_bearing_y)},
            .line_width    = Width{max_width * static_cast<int>(char_palette->size())}
        };
    }

    [[nodiscard]] auto getCharMetrics(Index c) const -> CharMetrics
    {
        FT_GlyphSlot slot = m_face->glyph;
        auto error = FT_Load_Char(
            m_face,
            Index_<FT_ULong>{static_cast<FT_ULong>(c)},
            FT_LOAD_RENDER
        );
        // TODO: Handle error

        return CharMetrics{
            .width     = Width(slot->bitmap.width),
            .height    = Height{static_cast<int>(slot->bitmap.rows)},
            .bearing_x = BearingX{/* 26.6 fixed point */ slot->metrics.horiBearingX >> 6}, 
            .bearing_y = BearingY{/* 26.6 fixed point */ slot->metrics.horiBearingY >> 6}
        };
    }

    [[nodiscard]] auto loadFace(FilenameView filename, FontSize height) noexcept -> FT_Error
    {
        constexpr auto first_face = 0;
        auto error = FT_New_Face(
            m_freetype.handle, // Static handle to freetype library
            filename->data(),
            Index_<FT_Long>{first_face},
            &m_face
        );
        if (error) return error;

        constexpr auto dummy = 0;
        constexpr auto auto_calculated = 0;
        error = FT_Set_Char_Size(
            m_face,
            Width_<FT_F26Dot6>{auto_calculated},
            Height_<FT_F26Dot6>{static_cast<FT_F26Dot6>(height << 6)},
            HorizontalResolution{dummy},
            VerticalResolution{dummy}
        );
        return error;
    }
};

#endif
