// -*-c++-*-

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Notify>
#include <osg/StateSet>
#include <osg/TriangleIndexFunctor>

#include <osgDB/FileNameUtils>
#include <osgDB/Registry>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    struct TriangleCollector
    {
        std::vector<unsigned int> indices;

        void operator()(unsigned int i0, unsigned int i1, unsigned int i2)
        {
            indices.push_back(i0);
            indices.push_back(i1);
            indices.push_back(i2);
        }
    };

    struct Accessor
    {
        int bufferView = -1;
        int componentType = 5126;
        size_t count = 0;
        std::string type;
        osg::Vec3 min;
        osg::Vec3 max;
        bool hasMinMax = false;
    };

    struct BufferView
    {
        size_t byteOffset = 0;
        size_t byteLength = 0;
        int target = 0;
    };

    struct Primitive
    {
        int positionAccessor = -1;
        int normalAccessor = -1;
        int texCoordAccessor = -1;
        int colorAccessor = -1;
        int indexAccessor = -1;
        int material = 0;
    };

    struct MaterialInfo
    {
        osg::Vec4 baseColor;
    };

    static size_t align4(size_t value)
    {
        return (value + 3u) & ~size_t(3u);
    }

    static std::string jsonEscape(const std::string& value)
    {
        std::ostringstream out;
        for (char c : value)
        {
            switch (c)
            {
                case '"': out << "\\\""; break;
                case '\\': out << "\\\\"; break;
                case '\n': out << "\\n"; break;
                case '\r': out << "\\r"; break;
                case '\t': out << "\\t"; break;
                default: out << c; break;
            }
        }
        return out.str();
    }

    static void appendBytes(std::vector<unsigned char>& buffer, const void* data, size_t size)
    {
        const unsigned char* first = static_cast<const unsigned char*>(data);
        buffer.insert(buffer.end(), first, first + size);
    }

    static void appendUInt32(std::vector<unsigned char>& buffer, std::uint32_t value)
    {
        appendBytes(buffer, &value, sizeof(value));
    }

    static void padBinary(std::vector<unsigned char>& buffer)
    {
        while ((buffer.size() % 4u) != 0u) buffer.push_back(0);
    }

    static void writeUInt32(std::ofstream& out, std::uint32_t value)
    {
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }

    static osg::Vec3 transformPoint(const osg::Matrix& matrix, const osg::Vec3& point)
    {
        return point * matrix;
    }

    static osg::Vec3 transformNormal(const osg::Matrix& matrix, const osg::Vec3& normal)
    {
        osg::Matrix normalMatrix = osg::Matrix::inverse(matrix);
        normalMatrix.transpose(normalMatrix);
        osg::Vec3 transformed(
            normal.x() * normalMatrix(0, 0) + normal.y() * normalMatrix(1, 0) + normal.z() * normalMatrix(2, 0),
            normal.x() * normalMatrix(0, 1) + normal.y() * normalMatrix(1, 1) + normal.z() * normalMatrix(2, 1),
            normal.x() * normalMatrix(0, 2) + normal.y() * normalMatrix(1, 2) + normal.z() * normalMatrix(2, 2));
        transformed.normalize();
        return transformed;
    }

    static bool almostEqual(float left, float right)
    {
        return std::fabs(left - right) < 0.0001f;
    }

    static bool sameColor(const osg::Vec4& left, const osg::Vec4& right)
    {
        return almostEqual(left.r(), right.r()) &&
               almostEqual(left.g(), right.g()) &&
               almostEqual(left.b(), right.b()) &&
               almostEqual(left.a(), right.a());
    }

    static osg::Vec4 colorFromStateSet(const osg::StateSet* stateSet, const osg::Vec4& fallback)
    {
        if (!stateSet)
        {
            return fallback;
        }

        const osg::Material* material = dynamic_cast<const osg::Material*>(stateSet->getAttribute(osg::StateAttribute::MATERIAL));
        if (!material)
        {
            return fallback;
        }

        return material->getDiffuse(osg::Material::FRONT);
    }

    static osg::Vec4 colorFromGeometry(const osg::Geometry& geometry, const osg::Vec4& fallback)
    {
        const osg::Array* colorArray = geometry.getColorArray();
        if (!colorArray || colorArray->empty() || colorArray->getBinding() != osg::Array::BIND_OVERALL)
        {
            return fallback;
        }

        if (const osg::Vec4Array* colors = dynamic_cast<const osg::Vec4Array*>(colorArray))
        {
            return colors->front();
        }
        if (const osg::Vec3Array* colors = dynamic_cast<const osg::Vec3Array*>(colorArray))
        {
            const osg::Vec3& color = colors->front();
            return osg::Vec4(color.r(), color.g(), color.b(), fallback.a());
        }
        return fallback;
    }

    class GLTFBuilder
    {
    public:
        bool addGeometry(const osg::Geometry& geometry, const osg::Matrix& matrix, const osg::Vec4& materialColor)
        {
            const osg::Vec3Array* vertices = dynamic_cast<const osg::Vec3Array*>(geometry.getVertexArray());
            if (!vertices || vertices->empty())
            {
                return false;
            }

            TriangleCollector collector;
            osg::TriangleIndexFunctor<TriangleCollector> triangleFunctor;
            geometry.accept(triangleFunctor);
            collector.indices.swap(triangleFunctor.indices);
            if (collector.indices.empty())
            {
                return false;
            }

            const osg::Vec3Array* sourceNormals = dynamic_cast<const osg::Vec3Array*>(geometry.getNormalArray());
            const bool hasVertexNormals =
                sourceNormals &&
                sourceNormals->size() >= vertices->size() &&
                sourceNormals->getBinding() == osg::Array::BIND_PER_VERTEX;

            const osg::Vec2Array* sourceTexCoords = dynamic_cast<const osg::Vec2Array*>(geometry.getTexCoordArray(0));
            const bool hasTexCoords = sourceTexCoords && sourceTexCoords->size() >= vertices->size();

            const osg::Array* sourceColors = geometry.getColorArray();
            const osg::Vec4Array* sourceVec4Colors = dynamic_cast<const osg::Vec4Array*>(sourceColors);
            const osg::Vec3Array* sourceVec3Colors = dynamic_cast<const osg::Vec3Array*>(sourceColors);
            const bool hasVertexColors =
                sourceColors &&
                sourceColors->getBinding() == osg::Array::BIND_PER_VERTEX &&
                sourceColors->size() >= vertices->size() &&
                (sourceVec4Colors || sourceVec3Colors);

            std::vector<float> positions;
            std::vector<float> normals;
            std::vector<float> texCoords;
            std::vector<float> colors;
            std::vector<std::uint32_t> indices;

            positions.reserve(collector.indices.size() * 3);
            if (hasVertexNormals) normals.reserve(collector.indices.size() * 3);
            if (hasTexCoords) texCoords.reserve(collector.indices.size() * 2);
            if (hasVertexColors) colors.reserve(collector.indices.size() * 4);
            indices.reserve(collector.indices.size());

            osg::Vec3 boundsMin(FLT_MAX, FLT_MAX, FLT_MAX);
            osg::Vec3 boundsMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (size_t i = 0; i < collector.indices.size(); ++i)
            {
                const unsigned int sourceIndex = collector.indices[i];
                if (sourceIndex >= vertices->size())
                {
                    continue;
                }

                const osg::Vec3 position = transformPoint(matrix, (*vertices)[sourceIndex]);
                positions.push_back(position.x());
                positions.push_back(position.y());
                positions.push_back(position.z());
                boundsMin.x() = std::min(boundsMin.x(), position.x());
                boundsMin.y() = std::min(boundsMin.y(), position.y());
                boundsMin.z() = std::min(boundsMin.z(), position.z());
                boundsMax.x() = std::max(boundsMax.x(), position.x());
                boundsMax.y() = std::max(boundsMax.y(), position.y());
                boundsMax.z() = std::max(boundsMax.z(), position.z());

                if (hasVertexNormals)
                {
                    const osg::Vec3 normal = transformNormal(matrix, (*sourceNormals)[sourceIndex]);
                    normals.push_back(normal.x());
                    normals.push_back(normal.y());
                    normals.push_back(normal.z());
                }

                if (hasTexCoords)
                {
                    const osg::Vec2 texCoord = (*sourceTexCoords)[sourceIndex];
                    texCoords.push_back(texCoord.x());
                    texCoords.push_back(texCoord.y());
                }

                if (hasVertexColors)
                {
                    osg::Vec4 color;
                    if (sourceVec4Colors)
                    {
                        color = (*sourceVec4Colors)[sourceIndex];
                    }
                    else
                    {
                        const osg::Vec3 vec3Color = (*sourceVec3Colors)[sourceIndex];
                        color = osg::Vec4(vec3Color.r(), vec3Color.g(), vec3Color.b(), materialColor.a());
                    }
                    colors.push_back(color.r());
                    colors.push_back(color.g());
                    colors.push_back(color.b());
                    colors.push_back(color.a());
                }

                indices.push_back(static_cast<std::uint32_t>(indices.size()));
            }

            if (indices.empty())
            {
                return false;
            }

            Primitive primitive;
            primitive.positionAccessor = addFloatAccessor(positions, "VEC3", 34962, true, boundsMin, boundsMax);
            if (!normals.empty())
            {
                primitive.normalAccessor = addFloatAccessor(normals, "VEC3", 34962, false, osg::Vec3(), osg::Vec3());
            }
            if (!texCoords.empty())
            {
                primitive.texCoordAccessor = addFloatAccessor(texCoords, "VEC2", 34962, false, osg::Vec3(), osg::Vec3());
            }
            if (!colors.empty())
            {
                primitive.colorAccessor = addFloatAccessor(colors, "VEC4", 34962, false, osg::Vec3(), osg::Vec3());
            }
            primitive.indexAccessor = addUIntAccessor(indices);
            primitive.material = addMaterial(colorFromGeometry(geometry, materialColor));
            _primitives.push_back(primitive);
            return true;
        }

        bool empty() const
        {
            return _primitives.empty();
        }

        bool write(const std::string& fileName) const
        {
            const std::string extension = osgDB::getLowerCaseFileExtension(fileName);
            if (extension == "glb")
            {
                return writeGlb(fileName);
            }
            return writeGltf(fileName);
        }

    private:
        int addBufferView(size_t offset, size_t length, int target)
        {
            BufferView view;
            view.byteOffset = offset;
            view.byteLength = length;
            view.target = target;
            _bufferViews.push_back(view);
            return static_cast<int>(_bufferViews.size() - 1);
        }

        int addFloatAccessor(const std::vector<float>& values, const std::string& type, int target, bool minMax, const osg::Vec3& boundsMin, const osg::Vec3& boundsMax)
        {
            padBinary(_binary);
            const size_t offset = _binary.size();
            appendBytes(_binary, values.data(), values.size() * sizeof(float));
            const int bufferView = addBufferView(offset, values.size() * sizeof(float), target);

            Accessor accessor;
            accessor.bufferView = bufferView;
            accessor.componentType = 5126;
            accessor.type = type;
            accessor.count = values.size() / componentCount(type);
            accessor.hasMinMax = minMax;
            accessor.min = boundsMin;
            accessor.max = boundsMax;
            _accessors.push_back(accessor);
            return static_cast<int>(_accessors.size() - 1);
        }

        int addMaterial(const osg::Vec4& baseColor)
        {
            for (size_t i = 0; i < _materials.size(); ++i)
            {
                if (sameColor(_materials[i].baseColor, baseColor))
                {
                    return static_cast<int>(i);
                }
            }

            MaterialInfo material;
            material.baseColor = baseColor;
            _materials.push_back(material);
            return static_cast<int>(_materials.size() - 1);
        }

        static size_t componentCount(const std::string& type)
        {
            if (type == "SCALAR") return 1;
            if (type == "VEC2") return 2;
            if (type == "VEC4") return 4;
            return 3;
        }

        int addUIntAccessor(const std::vector<std::uint32_t>& values)
        {
            padBinary(_binary);
            const size_t offset = _binary.size();
            appendBytes(_binary, values.data(), values.size() * sizeof(std::uint32_t));
            const int bufferView = addBufferView(offset, values.size() * sizeof(std::uint32_t), 34963);

            Accessor accessor;
            accessor.bufferView = bufferView;
            accessor.componentType = 5125;
            accessor.type = "SCALAR";
            accessor.count = values.size();
            _accessors.push_back(accessor);
            return static_cast<int>(_accessors.size() - 1);
        }

        std::string jsonDocument(const std::string& bufferUri) const
        {
            std::ostringstream json;
            json << std::setprecision(std::numeric_limits<float>::digits10 + 2);
            json << "{";
            json << "\"asset\":{\"version\":\"2.0\",\"generator\":\"OpenSceneGraph osgdb_gltf\"},";
            json << "\"scene\":0,";
            json << "\"scenes\":[{\"nodes\":[0]}],";
            json << "\"nodes\":[{\"name\":\"OSG Root\",\"mesh\":0}],";
            json << "\"meshes\":[{\"name\":\"OSG Mesh\",\"primitives\":[";
            for (size_t i = 0; i < _primitives.size(); ++i)
            {
                const Primitive& primitive = _primitives[i];
                if (i > 0) json << ",";
                json << "{\"attributes\":{\"POSITION\":" << primitive.positionAccessor;
                if (primitive.normalAccessor >= 0) json << ",\"NORMAL\":" << primitive.normalAccessor;
                if (primitive.texCoordAccessor >= 0) json << ",\"TEXCOORD_0\":" << primitive.texCoordAccessor;
                if (primitive.colorAccessor >= 0) json << ",\"COLOR_0\":" << primitive.colorAccessor;
                json << "},\"indices\":" << primitive.indexAccessor << ",\"material\":" << primitive.material << "}";
            }
            json << "]}],";
            json << "\"materials\":[";
            for (size_t i = 0; i < _materials.size(); ++i)
            {
                const osg::Vec4& color = _materials[i].baseColor;
                if (i > 0) json << ",";
                json << "{\"name\":\"Material " << i << "\",\"pbrMetallicRoughness\":{\"baseColorFactor\":["
                     << color.r() << "," << color.g() << "," << color.b() << "," << color.a()
                     << "],\"metallicFactor\":0.0,\"roughnessFactor\":0.85}}";
            }
            json << "],";
            json << "\"buffers\":[{\"byteLength\":" << _binary.size();
            if (!bufferUri.empty()) json << ",\"uri\":\"" << jsonEscape(bufferUri) << "\"";
            json << "}],";
            json << "\"bufferViews\":[";
            for (size_t i = 0; i < _bufferViews.size(); ++i)
            {
                const BufferView& view = _bufferViews[i];
                if (i > 0) json << ",";
                json << "{\"buffer\":0,\"byteOffset\":" << view.byteOffset << ",\"byteLength\":" << view.byteLength;
                if (view.target != 0) json << ",\"target\":" << view.target;
                json << "}";
            }
            json << "],";
            json << "\"accessors\":[";
            for (size_t i = 0; i < _accessors.size(); ++i)
            {
                const Accessor& accessor = _accessors[i];
                if (i > 0) json << ",";
                json << "{\"bufferView\":" << accessor.bufferView << ",\"componentType\":" << accessor.componentType
                     << ",\"count\":" << accessor.count << ",\"type\":\"" << accessor.type << "\"";
                if (accessor.hasMinMax)
                {
                    json << ",\"min\":[" << accessor.min.x() << "," << accessor.min.y() << "," << accessor.min.z() << "]";
                    json << ",\"max\":[" << accessor.max.x() << "," << accessor.max.y() << "," << accessor.max.z() << "]";
                }
                json << "}";
            }
            json << "]";
            json << "}";
            return json.str();
        }

        bool writeGlb(const std::string& fileName) const
        {
            std::string json = jsonDocument("");
            const size_t jsonPaddedLength = align4(json.size());
            json.resize(jsonPaddedLength, ' ');

            std::vector<unsigned char> binary = _binary;
            padBinary(binary);

            const std::uint32_t totalLength = static_cast<std::uint32_t>(12 + 8 + json.size() + 8 + binary.size());

            std::ofstream out(fileName.c_str(), std::ios::binary);
            if (!out) return false;
            writeUInt32(out, 0x46546C67u);
            writeUInt32(out, 2u);
            writeUInt32(out, totalLength);
            writeUInt32(out, static_cast<std::uint32_t>(json.size()));
            writeUInt32(out, 0x4E4F534Au);
            out.write(json.data(), json.size());
            writeUInt32(out, static_cast<std::uint32_t>(binary.size()));
            writeUInt32(out, 0x004E4942u);
            out.write(reinterpret_cast<const char*>(binary.data()), binary.size());
            return static_cast<bool>(out);
        }

        bool writeGltf(const std::string& fileName) const
        {
            const std::string binFileName = osgDB::getNameLessExtension(fileName) + ".bin";
            const std::string binSimpleName = osgDB::getSimpleFileName(binFileName);
            std::vector<unsigned char> binary = _binary;
            padBinary(binary);

            std::ofstream bin(binFileName.c_str(), std::ios::binary);
            if (!bin) return false;
            bin.write(reinterpret_cast<const char*>(binary.data()), binary.size());
            bin.close();

            std::ofstream gltf(fileName.c_str());
            if (!gltf) return false;
            gltf << jsonDocument(binSimpleName);
            return static_cast<bool>(gltf);
        }

        mutable std::vector<unsigned char> _binary;
        mutable std::vector<BufferView> _bufferViews;
        mutable std::vector<Accessor> _accessors;
        std::vector<MaterialInfo> _materials;
        std::vector<Primitive> _primitives;
    };

    class CollectGeometryVisitor : public osg::NodeVisitor
    {
    public:
        CollectGeometryVisitor(GLTFBuilder& builder)
            : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN),
              _builder(builder)
        {
            _matrixStack.push_back(osg::Matrix::identity());
            _colorStack.push_back(osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f));
        }

        virtual void apply(osg::MatrixTransform& transform)
        {
            _matrixStack.push_back(transform.getMatrix() * _matrixStack.back());
            _colorStack.push_back(colorFromStateSet(transform.getStateSet(), _colorStack.back()));
            traverse(transform);
            _colorStack.pop_back();
            _matrixStack.pop_back();
        }

        virtual void apply(osg::Geode& geode)
        {
            _colorStack.push_back(colorFromStateSet(geode.getStateSet(), _colorStack.back()));
            for (unsigned int i = 0; i < geode.getNumDrawables(); ++i)
            {
                const osg::Geometry* geometry = dynamic_cast<const osg::Geometry*>(geode.getDrawable(i));
                if (geometry)
                {
                    const osg::Vec4 color = colorFromStateSet(geometry->getStateSet(), _colorStack.back());
                    _builder.addGeometry(*geometry, _matrixStack.back(), color);
                }
            }
            traverse(geode);
            _colorStack.pop_back();
        }

    private:
        GLTFBuilder& _builder;
        std::vector<osg::Matrix> _matrixStack;
        std::vector<osg::Vec4> _colorStack;
    };
}

class ReaderWriterGLTF : public osgDB::ReaderWriter
{
public:
    ReaderWriterGLTF()
    {
        supportsExtension("gltf", "glTF 2.0 JSON scene format");
        supportsExtension("glb", "glTF 2.0 binary scene format");
    }

    virtual const char* className() const
    {
        return "glTF 2.0 Writer";
    }

    virtual WriteResult writeObject(const osg::Object& object, const std::string& fileName, const Options* options = NULL) const
    {
        const osg::Node* node = dynamic_cast<const osg::Node*>(&object);
        if (!node) return WriteResult(WriteResult::FILE_NOT_HANDLED);
        return writeNode(*node, fileName, options);
    }

    virtual WriteResult writeNode(const osg::Node& node, const std::string& fileName, const Options* /*options*/ = NULL) const
    {
        if (!acceptsExtension(osgDB::getFileExtension(fileName)))
        {
            return WriteResult(WriteResult::FILE_NOT_HANDLED);
        }

        GLTFBuilder builder;
        CollectGeometryVisitor visitor(builder);
        const_cast<osg::Node&>(node).accept(visitor);

        if (builder.empty())
        {
            return WriteResult("No triangle geometry found to write");
        }

        if (!builder.write(fileName))
        {
            return WriteResult(WriteResult::ERROR_IN_WRITING_FILE);
        }

        return WriteResult(WriteResult::FILE_SAVED);
    }
};

#if defined(OSGDB_GLB_PLUGIN)
extern "C" void osgdb_glb(void) {}
#else
extern "C" void osgdb_gltf(void) {}
#endif

static osgDB::RegisterReaderWriterProxy<ReaderWriterGLTF> g_readerWriterGLTFProxy;
