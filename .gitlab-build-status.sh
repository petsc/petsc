#!/usr/bin/env sh

# Push GitLab CI/CD build status to Bitbucket Cloud

if [ -z "$BITBUCKET_ACCESS_TOKEN" ]; then
   echo "ERROR: BITBUCKET_ACCESS_TOKEN is not set"
exit 1
fi
if [ -z "$BITBUCKET_USERNAME" ]; then
    echo "ERROR: BITBUCKET_USERNAME is not set"
exit 1
fi
if [ -z "$BITBUCKET_NAMESPACE" ]; then
    echo "Setting BITBUCKET_NAMESPACE to $CI_PROJECT_NAMESPACE"
    BITBUCKET_NAMESPACE=$CI_PROJECT_NAMESPACE
fi
if [ -z "$BITBUCKET_REPOSITORY" ]; then
    echo "Setting BITBUCKET_REPOSITORY to $CI_PROJECT_NAME"
    BITBUCKET_REPOSITORY=$CI_PROJECT_NAME
fi

BITBUCKET_API_ROOT="https://api.bitbucket.org/2.0"
BITBUCKET_STATUS_API="$BITBUCKET_API_ROOT/repositories/$BITBUCKET_NAMESPACE/$BITBUCKET_REPOSITORY/commit/$CI_COMMIT_SHA/statuses/build"
BITBUCKET_KEY="ci/gitlab-ci/$CI_JOB_NAME"

case "$BUILD_STATUS" in
running)
   BITBUCKET_STATE="INPROGRESS"
   BITBUCKET_DESCRIPTION="The build is running!"
   ;;
passed)
   BITBUCKET_STATE="SUCCESSFUL"
   BITBUCKET_DESCRIPTION="The build passed!"
   ;;
failed)
   BITBUCKET_STATE="FAILED"
   BITBUCKET_DESCRIPTION="The build failed."
   ;;
esac

echo "Pushing status to $BITBUCKET_STATUS_API..."
curl --request POST $BITBUCKET_STATUS_API \
--user $BITBUCKET_USERNAME:$BITBUCKET_ACCESS_TOKEN \
--header "Content-Type:application/json" \
--silent \
--data "{ \"state\": \"$BITBUCKET_STATE\", \"key\": \"$BITBUCKET_KEY\", \"description\":
\"$BITBUCKET_DESCRIPTION\",\"url\": \"$CI_PROJECT_URL/pipelines/$CI_PIPELINE_ID\" }"
