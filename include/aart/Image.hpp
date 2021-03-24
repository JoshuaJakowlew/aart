#ifndef AART_IMAGE_H
#define AART_IMAGE_H

#include <opencv2/opencv.hpp>

#include <aart/IResource.hpp>

class Image final : public IResource<Image, cv::Mat>
{
public:
    Image(resource_t&& resource) :
        m_resource{std::move(resource)}
    {}

    Image(Filename const& filename)
    {
        read(filename);
    }

    auto read(Filename const& filename) -> resource_t&
    {
        m_resource = cv::imread(filename);
        return m_resource;
    }

    auto write(Filename const& filename) const -> bool
    {
        return cv::imwrite(filename, m_resource);
    }

    [[nodiscard]] auto get() -> resource_t&
    {
        return m_resource;
    }
    [[nodiscard]] auto get() const -> const resource_t&
    {
        return m_resource;
    }

    auto assign(resource_t&& resource) -> bool
    {
        m_resource = std::move(resource);
        return true;
    }
private:
    resource_t m_resource;
};

#endif
