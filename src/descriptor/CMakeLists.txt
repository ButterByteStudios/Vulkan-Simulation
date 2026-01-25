add_library(DescriptorLib STATIC
	"descriptorLayoutBuilder.cpp"
	"descriptorAllocator.cpp")

target_include_directories(DescriptorLib
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(DescriptorLib
	PRIVATE 
		Vulkan::Vulkan
)