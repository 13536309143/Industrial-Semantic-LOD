#include "meshlod.cpp"
#include <cstdio>
#include <stdexcept>
#include <random>

static void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Capture { std::vector<std::vector<unsigned int>> leaves; unsigned int locked; bool lost=false; };
static int output(void* p, clodGroup, const clodCluster* clusters, size_t count, size_t, unsigned int)
{
    auto& c=*static_cast<Capture*>(p);
    for(size_t i=0;i<count;++i)
        if(clusters[i].refined<0)c.leaves.emplace_back(clusters[i].indices,clusters[i].indices+clusters[i].index_count);
    return int(c.leaves.size());
}

static void testCircularOpenings()
{
    const unsigned int n=128;
    std::vector<float> original;
    std::vector<unsigned int> faces;
    for(float radius : {1.f,2.f})
        for(unsigned int i=0;i<n;++i)
        {float a=float(i)*6.283185307179586f/float(n);original.insert(original.end(),{radius*cosf(a),radius*sinf(a),0.f});}
    for(unsigned int i=0;i<n;++i)
    {unsigned int j=(i+1)%n;faces.insert(faces.end(),{i,j,n+j,i,n+j,n+i});}
    for(float scale : {.001f,1.f,1000.f})
    {
        std::vector<float> vertices=original;
        for(size_t i=0;i<vertices.size();i+=3)
        {float x=vertices[i],y=vertices[i+1];vertices[i]=scale*(.8f*x+.37f);vertices[i+1]=scale*(+.6f*x*.6f+.8f*y-.23f);vertices[i+2]=scale*(-.6f*x*.8f+.6f*y+.51f);}
        clod::IterationContext c;c.config=clodDefaultConfig(128);c.config.feature_hard_lock_ratio=.15f;
        c.mesh={};c.mesh.indices=faces.data();c.mesh.index_count=faces.size();
        c.mesh.vertex_positions=vertices.data();c.mesh.vertex_positions_stride=12;c.mesh.vertex_count=vertices.size()/3;
        clodFeatureMetrics metrics{};c.mesh.feature_metrics=&metrics;
        c.remap.resize(c.mesh.vertex_count);meshopt_generatePositionRemap(c.remap.data(),vertices.data(),c.mesh.vertex_count,12);
        clod::analyzeFeatureConstraints(c);
        require(metrics.circular_hole_loops==1,"annulus opening must be recognized; filled outer disk must be rejected");
        require(metrics.critical_vertices==12,"opening anchors must be deterministic across rotation and scale");
        {
            const auto curves=clod::featureCurveSegments(c.mesh,faces);
            std::vector<unsigned char> caller(c.mesh.vertex_count,0);caller[n]=meshopt_SimplifyVertex_Lock;
            const auto whole=clod::featureWholeRimLocks(c.mesh,faces,curves,caller,128,.25f);
            require(!whole.empty()&&(whole[n]&meshopt_SimplifyVertex_Lock),"whole-rim proposal erased an explicit caller lock");
            require(std::count_if(whole.begin(),whole.end(),[](unsigned char f){return (f&meshopt_SimplifyVertex_Lock)!=0;})==25,"whole-rim proposal must cover the full opening with24 arc-length anchors plus caller lock");
            require(clod::featureWholeRimLocks(c.mesh,faces,curves,caller,16,.25f).empty(),"insufficient ring budget must skip the proposal rather than partially lock a hole");
            require(clod::featureWholeRimLocks(c.mesh,faces,curves,caller,128,0.f).empty(),"zero whole-rim budget must disable the proposal");
        }
        std::vector<float> anchor_angles;
        for(unsigned int i=0;i<n;++i)
            if(c.feature_locks[i]&meshopt_SimplifyVertex_Lock)anchor_angles.push_back(float(i)*6.283185307179586f/float(n));
        require(anchor_angles.size()==12,"the circular opening must receive its allocated anchors");
        float max_gap=0;
        for(size_t i=0;i<anchor_angles.size();++i)
        {float next=i+1<anchor_angles.size()?anchor_angles[i+1]:anchor_angles[0]+6.283185307179586f;max_gap=std::max(max_gap,next-anchor_angles[i]);}
        require(max_gap<.55f,"circular anchors must avoid the uneven gaps of a truncated farthest-point sequence");
        require(std::any_of(c.feature_locks.begin(),c.feature_locks.end(),[](unsigned char f){return (f&meshopt_SimplifyVertex_Lock)!=0;}),"geometric anchors must use genuine Lock, with optional Protect seam flags");
    }
    std::vector<float> disk={0,0,0};std::vector<unsigned int> triangles;
    for(unsigned int i=0;i<n;++i)
    {float a=float(i)*6.283185307179586f/float(n);disk.insert(disk.end(),{cosf(a),sinf(a),0.f});triangles.insert(triangles.end(),{0,1+i,1+(i+1)%n});}
    clod::IterationContext c;c.config=clodDefaultConfig(128);
    c.mesh={};c.mesh.indices=triangles.data();c.mesh.index_count=triangles.size();
    c.mesh.vertex_positions=disk.data();c.mesh.vertex_positions_stride=12;c.mesh.vertex_count=disk.size()/3;
    clodFeatureMetrics metrics{};c.mesh.feature_metrics=&metrics;
    c.remap.resize(c.mesh.vertex_count);meshopt_generatePositionRemap(c.remap.data(),disk.data(),c.mesh.vertex_count,12);
    clod::analyzeFeatureConstraints(c);
    require(metrics.circular_hole_loops==0&&metrics.critical_vertices==0,"filled circular caps must not become hard-locked holes");
}

int main()
{
    try
    {
        testCircularOpenings();
        {
            std::vector<float> split={0,0,0,1,0,0,1,1,0,0,0,0,1,1,0,0,1,0};
            std::vector<unsigned int> triangles={0,1,2,3,4,5};
            clodMesh mesh{};mesh.vertex_positions=split.data();mesh.vertex_count=6;mesh.vertex_positions_stride=12;
            auto curves=clod::featureCurveSegments(mesh,triangles);
            require(curves.size()==12,"normal seam must not become a false geometric boundary");
            clod::FeatureSurfaceTree tree(mesh,curves);double tail=1;
            require(clod::featureCurveError(mesh,curves,tree,&tail)<1e-12&&tail<1e-12,"identical complete curves must have zero mean and tail error");
        }
        float a[]={0,0,0},b[]={1,0,0},c[]={0,1,0};
        float p[]={.2f,.2f,1};
        require(fabsf(clod::featurePointTriangleDistanceSq(p,a,b,c)-1)<1e-5f,"point/triangle interior distance");
        float q[]={1,1,0};
        require(fabsf(clod::featurePointTriangleDistanceSq(q,a,b,c)-.5f)<1e-5f,"point/triangle edge distance");
        float d[]={2,0,0},r[]={.5f,1,0};
        require(fabsf(clod::featurePointTriangleDistanceSq(r,a,b,d)-1)<1e-5f,"degenerate triangle distance");
        std::vector<float> vertices;
        std::vector<unsigned int> faces;
        const unsigned int side=17;
        for(unsigned int y=0;y<side;++y)for(unsigned int x=0;x<side;++x)
            vertices.insert(vertices.end(),{float(x),float(y),0.f});
        for(unsigned int y=0;y+1<side;++y)for(unsigned int x=0;x+1<side;++x)
        {unsigned int k=y*side+x;faces.insert(faces.end(),{k,k+1,k+side+1,k,k+side+1,k+side});}
        // A flat rectangular panel has no circular hole: its four genuine
        // boundary corners must still become anchors and survive the guard.
        {
            clod::IterationContext local;local.config=clodDefaultConfig(128);
            local.mesh={};local.mesh.indices=faces.data();local.mesh.index_count=faces.size();
            local.mesh.vertex_positions=vertices.data();local.mesh.vertex_positions_stride=12;local.mesh.vertex_count=vertices.size()/3;
            clodFeatureMetrics metrics{};local.mesh.feature_metrics=&metrics;
            local.remap.resize(local.mesh.vertex_count);meshopt_generatePositionRemap(local.remap.data(),vertices.data(),local.mesh.vertex_count,12);
            clod::analyzeFeatureConstraints(local);
            require(metrics.circular_hole_loops==0&&metrics.critical_vertices==4,"noncircular panel must protect four boundary corners");
            std::vector<unsigned char> borders(local.mesh.vertex_count,0);float e=0;
            auto candidate=clod::simplify(local.config,local.mesh,faces,borders,faces.size()/16,&e);
            for(unsigned int v=0;v<local.mesh.vertex_count;++v)if(local.feature_locks[v]&meshopt_SimplifyVertex_Lock)
                require(std::find(candidate.begin(),candidate.end(),v)!=candidate.end(),"quality guard erased a detected boundary anchor");
        }
        const unsigned int locked=8*side+8;
        {
            clodMesh mesh{};mesh.vertex_positions=vertices.data();mesh.vertex_positions_stride=12;mesh.vertex_count=vertices.size()/3;
            clod::FeatureSurfaceTree tree(mesh,faces);
            std::mt19937 generator(73419);std::uniform_real_distribution<float> sample(-3.f,20.f);
            for(int i=0;i<100;++i)
            {
                float point[3]={sample(generator),sample(generator),sample(generator)*.1f};float nearest=FLT_MAX,expected=FLT_MAX;size_t triangle=0;
                tree.query(point,0,nearest,triangle);
                for(size_t k=0;k<faces.size();k+=3)expected=std::min(expected,clod::featurePointTriangleDistanceSq(point,&vertices[faces[k]*3],&vertices[faces[k+1]*3],&vertices[faces[k+2]*3]));
                require(fabsf(nearest-expected)<1e-5f,"BVH nearest distance must equal exhaustive triangle search");
            }
            double normal=0;
            require(clod::featureQuadratureError(mesh,faces,tree,&normal)<1e-8&&normal<1e-8,"identical surfaces must have zero quadrature and normal error");
            auto reversed=faces;for(size_t k=0;k<reversed.size();k+=3)std::swap(reversed[k],reversed[k+1]);
            clod::FeatureSurfaceTree flipped(mesh,reversed);
            require(clod::featureQuadratureError(mesh,faces,flipped,&normal)<1e-8&&normal>1.999,"normal gate must detect reversed winding even when geometry coincides");
        }
        std::vector<unsigned char> locks(vertices.size()/3);locks[locked]=meshopt_SimplifyVertex_Lock;
        {
            clodConfig cfg=clodDefaultConfig(128);cfg.simplify_fallback_sloppy=false;
            clodMesh mesh{};mesh.indices=faces.data();mesh.index_count=faces.size();mesh.vertex_positions=vertices.data();mesh.vertex_positions_stride=12;mesh.vertex_count=locks.size();
            std::vector<unsigned char> automatic(locks.size(),meshopt_SimplifyVertex_Lock),empty(locks.size(),0);
            mesh.feature_lock=automatic.data();mesh.feature_user_lock=locks.data();
            float e=0;auto guarded=clod::simplify(cfg,mesh,faces,empty,faces.size()/4,&e);
            require(guarded.size()<faces.size(),"adaptive automatic anchors must not force a globally harmful unreduced mesh");
            require(std::find(guarded.begin(),guarded.end(),locked)!=guarded.end(),"quality gate must retain caller engineering locks");
            clodMesh plain=mesh;plain.feature_lock=nullptr;plain.feature_user_lock=nullptr;
            clodConfig base=cfg;base.feature_constraints=false;
            float plain_error=0;auto baseline=clod::simplify(base,plain,faces,locks,faces.size()/4,&plain_error);
            require(guarded==baseline,"unhelpful automatic proposals must reproduce the true feature-off baseline");
            cfg.feature_quality_guard=false;
            auto strict=clod::simplify(cfg,mesh,faces,empty,faces.size()/4,&e);
            require(strict.size()==faces.size(),"explicit strict-anchor mode must retain its constraints");
        }
        for(int feature=0;feature<=1;++feature)
            for(int source=0;source<=1;++source)
            {
                clodConfig cfg=clodDefaultConfig(128);cfg.feature_constraints=bool(feature);
                cfg.base_simplify_ratio=.25f;cfg.base_error_relative=.1f;
                cfg.simplify_fallback_sloppy=false;
                clodMesh mesh{};mesh.indices=faces.data();mesh.index_count=faces.size();
                mesh.vertex_positions=vertices.data();mesh.vertex_positions_stride=12;mesh.vertex_count=locks.size();
                if(source==0)mesh.vertex_lock=locks.data();else mesh.feature_lock=locks.data();
                Capture cap{};cap.locked=locked;
                clodBuild(cfg,mesh,&cap,output,nullptr);
                bool found=false;
                for(const auto& leaf:cap.leaves)found|=std::find(leaf.begin(),leaf.end(),locked)!=leaf.end();
                require(found,"caller hard lock lost during base pre-simplification");
            }
        clod::IterationContext context;
        context.config=clodDefaultConfig(128);
        context.mesh={};context.mesh.indices=faces.data();context.mesh.index_count=faces.size();
        context.mesh.vertex_positions=vertices.data();context.mesh.vertex_positions_stride=12;
        context.mesh.vertex_count=locks.size();context.mesh.feature_lock=locks.data();
        std::vector<float> attributes(locks.size()*32,0.f),weights(32,.1f);
        context.mesh.attribute_count=32;context.mesh.vertex_attributes=attributes.data();
        context.mesh.vertex_attributes_stride=128;context.mesh.attribute_weights=weights.data();
        context.remap.resize(locks.size());meshopt_generatePositionRemap(context.remap.data(),vertices.data(),locks.size(),12);
        clod::analyzeFeatureConstraints(context);
        require(context.mesh.attribute_count==32,"feature channels exceed meshoptimizer 32-attribute limit");
        require(context.mesh.feature_attribute_count==0,"caller attributes must not be relabelled as feature channels");
        require(context.feature_locks[locked]&meshopt_SimplifyVertex_Lock,"caller feature lock overwritten by analyzer");
        std::puts("PASS: point/triangle distances, degenerate triangle, caller locks with base reduction, attribute limit, feature-lock merging");
        return 0;
    }
    catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
