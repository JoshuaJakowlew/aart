#ifndef AART_IMAGE_MANAGER_H
#define AART_IMAGE_MANAGER_H

#include <opencv2/opencv.hpp>

#include <aart/IResourceManager.hpp>

class ImageManager final : public IResourceManager<ImageManager, cv::Mat>
{
public:
    ImageManager() = default;
    ImageManager(ImageManager&&) = default;

    ImageManager(const std::string& filename)
    {
        read(filename);
    }

    [[nodiscard]] auto read(const std::string& filename) -> resource_t&
    {
        m_resource = cv::imread(filename);
        return m_resource;
    }

    [[nodiscard]] auto write(const std::string& filename) -> bool
    {
        return cv::imwrite(filename, m_resource);
    }

    [[nodiscard]] auto getResource() noexcept -> resource_t&
    {
        return m_resource;
    }

    [[nodiscard]] auto assign(resource_t&& resource) -> bool
    {
        m_resource = std::move(resource);
        return true;
    }
private:
    resource_t m_resource;
};

#endif
