import open3d as o3d
import os
import glob
from tqdm import tqdm

def sample_meshes_to_pcd(input_dir, output_dir, num_points=10000):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all .obj files in the input directory
    search_pattern = os.path.join(input_dir, "*.obj")
    obj_files = glob.glob(search_pattern)

    if not obj_files:
        print(f"No .obj files found in {input_dir}")
        return

    for obj_path in tqdm(obj_files):
        try:
            # Load the mesh
            mesh = o3d.io.read_triangle_mesh(obj_path)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            # Check if mesh is empty
            if not mesh.has_vertices():
                print(f"Warning: {obj_path} is empty or invalid. Skipping.")
                continue

            # Sample points uniformly from the mesh surface
            pcd = mesh.sample_points_uniformly(number_of_points=num_points, use_triangle_normal=True)

            # You can alternatively use Poisson disk sampling for a more even distribution:
            # pcd = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=5)

            # Construct the output filename
            base_name = os.path.basename(obj_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_filename = f"{file_name_without_ext}.pcd"
            output_path = os.path.join(output_dir, output_filename)

            # Save the point cloud
            o3d.io.write_point_cloud(output_path, pcd)
            #print(f"Processed: {base_name} -> {output_filename}")

        except Exception as e:
            print(f"Error processing {obj_path}: {e}")

# Example usage
if __name__ == "__main__":
    element_class = "pipe"
    INPUT_FOLDER = "../mesh_dataset/" + element_class + "/obj/"   # Replace with your input folder path
    OUTPUT_FOLDER = "../mesh_dataset/" + element_class + "/pcd/"  # Replace with your output folder path
    POINTS_TO_SAMPLE = 100000             # Adjust the sampling density as needed

    sample_meshes_to_pcd(INPUT_FOLDER, OUTPUT_FOLDER, num_points=POINTS_TO_SAMPLE)