#!/usr/bin/env groovy
def alljob = JOB_NAME.tokenize('/') as String[]
def node_name = alljob[0]
def arch_name = alljob[1]

pipeline { 
    agent { 
        node { 
            label node_name
            customWorkspace "${arch_name}/${BRANCH_NAME}"
        } 
    }
    
    stages {
        stage('Local Merge') {
            steps {
                echo "Current dir: ${pwd()}"
                echo "Workspace variable: ${WORKSPACE}"
                echo "Running on node: ${node_name}"
                echo "Building for arch: ${arch_name}"
                checkout scm
            }
        }
        stage('Configure') {
            steps {
                sh "./config/examples/${arch_name}.py"
            }
        }
        stage('Make') {
            steps {
                sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} all"
            }
        }
        stage('Examples') {
            steps {
                sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} cleantest allgtest-tap TIMEOUT=300"
            }
            post {
                always {
                    junit "**/${arch_name}/tests/testresults.xml"
                    sh "make PETSC_ARCH=${arch_name} PETSC_DIR=${WORKSPACE} cleantest"
                }
            }
        }
    }  
}
