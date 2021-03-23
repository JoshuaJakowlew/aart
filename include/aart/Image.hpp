#ifndef AART_IMAGE_H
#define AART_IMAGE_H

#include <opencv2/opencv.hpp>

#include <aart/IResource.hpp>

class Image final : public IResource<Image, cv::Mat>
{
public:
    Image() = default;
    Image(Image&&) = default;
    
    Image(resource_t&& resource) :
        m_resource{std::move(resource)}
    {}

    Image(const std::string& filename)
    {
        read(filename);
    }

    auto read(const std::string& filename) -> resource_t&
    {
        m_resource = cv::imread(filename);
        return m_resource;
    }

    auto write(const std::string& filename) -> bool
    {
        return cv::imwrite(filename, m_resource);
    }

    [[nodiscard]] auto get() noexcept -> resource_t&
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
